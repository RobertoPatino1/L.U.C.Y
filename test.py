import chainlit as cl
import time

from test3 import astuff

@cl.on_chat_start
async def start():
    id = await cl.Message(content="Started stuff").send()
    await cl.Message(content="Stuff 1 started", parent_id=id).send()
    value = await astuff(id)
    
    
