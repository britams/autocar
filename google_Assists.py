from pop.popAssist import *

stream = create_conversation_stream()
ga = GAssistant(stream)

print("Taking about ...")
ga.assist()

print("Bye...")
stream.close()