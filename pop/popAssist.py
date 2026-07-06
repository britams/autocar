from threading import Thread

import concurrent.futures
import json
import logging
import pathlib2 as pathlib
import sys
import uuid

import grpc
import google.auth.transport.grpc
import google.auth.transport.requests
import google.oauth2.credentials

import threading

from google.assistant.embedded.v1alpha2 import (
    embedded_assistant_pb2,
    embedded_assistant_pb2_grpc
)

from tenacity import retry, stop_after_attempt, retry_if_exception

try:
    from googlesamples.assistant.grpc import (
        assistant_helpers,
        audio_helpers,
        browser_helpers,
        device_helpers
    )
except (SystemError, ImportError):
    import assistant_helpers
    import audio_helpers
    import browser_helpers
    import device_helpers

ASSISTANT_API_ENDPOINT = 'embeddedassistant.googleapis.com'
END_OF_UTTERANCE = embedded_assistant_pb2.AssistResponse.END_OF_UTTERANCE
DIALOG_FOLLOW_ON = embedded_assistant_pb2.DialogStateOut.DIALOG_FOLLOW_ON
CLOSE_MICROPHONE = embedded_assistant_pb2.DialogStateOut.CLOSE_MICROPHONE
PLAYING = embedded_assistant_pb2.ScreenOutConfig.PLAYING
DEFAULT_GRPC_DEADLINE = 60 * 3 + 5

verbose = False
logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

def get_device_id():
    with open('/home/soda/.config/googlesamples-assistant/device_config.json') as f:
        device = json.load(f)
        return device['id']

def get_device_model_id():
    with open('/home/soda/.config/googlesamples-assistant/device_config.json') as f:
        device = json.load(f)
        return device['model_id']

class GAssistant():
    def __init__(self, conversation_stream=None, local_device_handler = None, google_device_handler=None, lang="ko-KR", display=False):
        self.conversation_stream = conversation_stream 
        self.local_device_handler = local_device_handler
        self.device_handler = google_device_handler
        self.display = display
        self.language_code = lang
        self.conversation_state = None
        self.is_new_conversation = True

        credentials='/home/soda/.config/google-oauthlib-tool/credentials.json'
        with open(credentials, 'r') as f:
            credentials = google.oauth2.credentials.Credentials(token=None, **json.load(f))
            http_request = google.auth.transport.requests.Request()
            credentials.refresh(http_request)

        api_endpoint = ASSISTANT_API_ENDPOINT
        grpc_channel = google.auth.transport.grpc.secure_authorized_channel(credentials, http_request, api_endpoint)
        logging.info('Connecting to %s', api_endpoint)

        self.device_id = get_device_id()
        self.device_model_id = get_device_model_id()
        logging.info("Using device model %s and device id %s", self.device_model_id, self.device_id)

        self.assistant = embedded_assistant_pb2_grpc.EmbeddedAssistantStub(grpc_channel)
        self.deadline = DEFAULT_GRPC_DEADLINE
    
    def __enter__(self):
        return self

    def __exit__(self, etype, e, traceback):
        if e:
            return False

        self.conversation_stream_close()

    def conversation_stream_close(self):
        self.conversation_stream.close()

    def get_device_id(self):
        return self.device_id

    def get_device_handler(self):
        return device_helpers.DeviceRequestHandler(self.device_id)

    def is_grpc_error_unavailable(e):
        is_grpc_error = isinstance(e, grpc.RpcError)
        if is_grpc_error and (e.code() == grpc.StatusCode.UNAVAILABLE):
            logging.error('grpc unavailable error: %s', e)
            return True
        return False

    @retry(reraise=True, stop=stop_after_attempt(3),
           retry=retry_if_exception(is_grpc_error_unavailable))
    def assist(self, rec_request_handler=None, resp_handler=None, conversation_stream=None):
        if conversation_stream:
            if self.conversation_stream:
                self.conversation_stream.close()
            self.conversation_stream = conversation_stream
        
        continue_conversation = False
        device_actions_futures = []

        self.conversation_stream.start_recording()
        logging.info('Recording audio request.')
        if rec_request_handler:
            threading.Thread(target=rec_request_handler).start()

        def iter_log_assist_requests():
            try:
                for c in self.gen_assist_requests():
                    assistant_helpers.log_assist_request_without_audio(c)
                    yield c
            except RuntimeError:
                pass
            logging.debug('Reached end of AssistRequest iteration.')

        last_resp = None 
        is_local_device_handler = False

        for resp in self.assistant.Assist(iter_log_assist_requests(), self.deadline):
            assistant_helpers.log_assist_response_without_audio(resp)
            if resp.event_type == END_OF_UTTERANCE:
                logging.info('End of audio request detected.')
                logging.info('Stopping recording.')
                self.conversation_stream.stop_recording()
                if last_resp and self.local_device_handler:
                    data = ' '.join(last_resp)
                    data = ' '.join(data.split())
                    is_local_device_handler = self.local_device_handler(data)
            if resp.speech_results: 
                last_resp = [r.transcript for r in resp.speech_results]
            if len(resp.audio_out.audio_data) > 0 and not is_local_device_handler:
                if not self.conversation_stream.playing:
                    self.conversation_stream.stop_recording()
                    self.conversation_stream.start_playback()
                    logging.info('Playing assistant response.')
                self.conversation_stream.write(resp.audio_out.audio_data)
            if resp.dialog_state_out.conversation_state:
                conversation_state = resp.dialog_state_out.conversation_state
                logging.debug('Updating conversation state.')
                self.conversation_state = conversation_state
            if resp.dialog_state_out.volume_percentage != 0:
                volume_percentage = resp.dialog_state_out.volume_percentage
                logging.info('Setting volume to %s%%', volume_percentage)
                self.conversation_stream.volume_percentage = volume_percentage
            if resp.dialog_state_out.microphone_mode == DIALOG_FOLLOW_ON:
                continue_conversation = True
                logging.info('Expecting follow-on query from user.')
            elif resp.dialog_state_out.microphone_mode == CLOSE_MICROPHONE:
                continue_conversation = False
            if resp.device_action.device_request_json:
                device_request = json.loads(resp.device_action.device_request_json)
                fs = self.device_handler(device_request)
                if fs:
                    device_actions_futures.extend(fs)
            if self.display and resp.screen_out.data:
                system_browser = browser_helpers.system_browser
                system_browser.display(resp.screen_out.data)
        
        if len(device_actions_futures):
            logging.info('Waiting for device executions to complete.')
            concurrent.futures.wait(device_actions_futures)

        logging.info('Finished playing assistant response.')
        self.conversation_stream.stop_playback()
        if resp_handler:
            threading.Thread(target=resp_handler).start()
        return continue_conversation

    def gen_assist_requests(self):
        """Yields: AssistRequest messages to send to the API."""

        config = embedded_assistant_pb2.AssistConfig(
            audio_in_config=embedded_assistant_pb2.AudioInConfig(
                encoding='LINEAR16',
                sample_rate_hertz=self.conversation_stream.sample_rate,
            ),
            audio_out_config=embedded_assistant_pb2.AudioOutConfig(
                encoding='LINEAR16',
                sample_rate_hertz=self.conversation_stream.sample_rate,
                volume_percentage=self.conversation_stream.volume_percentage,
            ),
            dialog_state_in=embedded_assistant_pb2.DialogStateIn(
                language_code=self.language_code,
                conversation_state=self.conversation_state,
                is_new_conversation=self.is_new_conversation,
            ),
            device_config=embedded_assistant_pb2.DeviceConfig(
                device_id=self.device_id,
                device_model_id=self.device_model_id,
            )
        )

        if self.display:
            config.screen_out_config.screen_mode = PLAYING

        self.is_new_conversation = False
        yield embedded_assistant_pb2.AssistRequest(config=config)
        for data in self.conversation_stream:
            yield embedded_assistant_pb2.AssistRequest(audio_in=data)


def create_conversation_stream(
    audio_sample_rate=audio_helpers.DEFAULT_AUDIO_SAMPLE_RATE, 
    audio_sample_width=audio_helpers.DEFAULT_AUDIO_SAMPLE_WIDTH, 
    audio_block_size=audio_helpers.DEFAULT_AUDIO_DEVICE_BLOCK_SIZE, 
    audio_flush_size=audio_helpers.DEFAULT_AUDIO_DEVICE_FLUSH_SIZE, 
    audio_iter_size=audio_helpers.DEFAULT_AUDIO_ITER_SIZE):

    audio_source = audio_helpers.SoundDeviceStream(
        sample_rate=audio_sample_rate,
        sample_width=audio_sample_width,
        block_size=audio_block_size,
        flush_size=audio_flush_size
    )

    stream = audio_helpers.ConversationStream(
        source=audio_source,
        sink=audio_source,
        iter_size=audio_iter_size,
        sample_width=audio_sample_width,
    )

    return stream

def create_device_handler():
    return device_helpers.DeviceRequestHandler(get_device_id())