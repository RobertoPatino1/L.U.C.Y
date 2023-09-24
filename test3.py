import chainlit as cl
import time

def another_stuff():
    time.sleep(10)
    return 0
    
aanother_stuff = cl.make_async(another_stuff)

def stuff(id):
    id_2 = cl.run_sync(cl.Message(content="Started stuff", parent_id= id).send())
    cl.run_sync(cl.Message(content="Stuff 2", parent_id=id_2).send())
    value = cl.run_sync(aanother_stuff())
    cl.run_sync(cl.Message(content="Stuff 3", parent_id=id_2).send())
    value = cl.run_sync(aanother_stuff())
    cl.run_sync(cl.Message(content="Already started stuff 3", parent_id=id_2).send())
    return 0

astuff = cl.make_async(stuff)
