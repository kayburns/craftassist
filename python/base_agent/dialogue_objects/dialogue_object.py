"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
import numpy as np
import random

from string_lists import MAP_YES, MAP_NO
from util import pos_to_np


class DialogueObject(object):
    def __init__(self, agent, memory, dialogue_stack, featurizer=None, max_steps=50):
        self.agent = agent
        self.memory = memory
        self.dialogue_stack = dialogue_stack
        self.featurizer = featurizer
        self.finished = False
        self.awaiting_response = False
        self.max_steps = max_steps  # finish after this many steps to avoid getting stuck
        self.current_step = 0
        self.progeny_data = (
            []
        )  # this should have more structure and some methods for adding/accessing?

    def step(self):
        raise NotImplementedError("Subclasses must implement step()")

    def update_progeny_data(self, data):
        self.progeny_data.append(data)

    def check_finished(self):
        """Check if the object is finished processing."""
        self.current_step += 1
        if self.current_step == self.max_steps:
            logging.error("Stepped {} {} times, finishing".format(self, self.max_steps))
            self.finished = True
        return self.finished

    def featurize(self):
        if self.featurizer is not None:
            return self.featurizer(self)
        else:
            return "empty"

    def __repr__(self):
        return str(type(self))


"""This class represents a sub-type of DialogueObject to await
a response from the user."""

# TODO check who is speaking
class AwaitResponse(DialogueObject):
    def __init__(self, wait_time=800, **kwargs):
        super().__init__(**kwargs)
        self.init_time = self.memory.get_time()
        self.response = []
        self.wait_time = wait_time
        self.awaiting_response = True

    def step(self):
        """Wait for wait_time for an answer. Mark finished when a chat comes in."""
        chatmem = self.memory.get_most_recent_incoming_chat(after=self.init_time + 1)
        if chatmem is not None:
            self.finished = True
            return "", {"response": chatmem}
        if self.memory.get_time() - self.init_time > self.wait_time:
            self.finished = True
            # FIXME this shouldn't return data
            return "Okay! I'll stop waiting for you to answer that.", {"response": None}
        return "", None


"""This class represents a sub-type of DialogueObject to say / send a chat
to the user."""


class Say(DialogueObject):
    def __init__(self, response_options, **kwargs):
        super().__init__(**kwargs)
        if len(response_options) == 0:
            raise ValueError("Cannot init a Say with no response options")

        if type(response_options) is str:
            self.response_options = [response_options]
        else:
            self.response_options = response_options

    def step(self):
        """Return one of the response_options."""
        self.finished = True
        return random.choice(self.response_options), None


"""This class represents a sub-type of the Say DialogueObject above to answer
something about the current capabilities of the bot, to the user."""


class BotCapabilities(Say):
    def __init__(self, **kwargs):
        response_options = [
            'Try looking at something and tell me "go there"',
            'Try looking at a structure and tell me "destroy that"',
            'Try looking somewhere and tell me "build a wall there"',
            "Try building something and giving it a name",
            "Try naming something and telling me to build it",
        ]
        super().__init__(response_options, **kwargs)


"""This class represents a sub-type of the Say DialogueObject above to greet
the user as a reply to a greeting."""


class BotGreet(Say):
    def __init__(self, **kwargs):
        response_options = ["hi there!", "hello", "hey", "hi"]
        super().__init__(response_options, **kwargs)


"""This class represents a sub-type of the DialogueObject to answer
questions about the current status of the bot, to the user."""


class BotStackStatus(DialogueObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ing_mapping = {
            "Build": "building",
            "Destroy": "excavating",
            "Dig": "digging",
            "Move": "moving",
        }

    def step(self):
        """return the current task name status."""
        self.finished = True

        task_mem = self.memory.task_stack_find_lowest_instance(list(self.ing_mapping.keys()))
        if task_mem is None:
            answer_options = [
                "Idle. You can tell me what to do!",
                "I am doing nothing.",
                "Nothing. Waiting for your command.",
            ]
        else:
            doing = self.ing_mapping[task_mem.task.__class__.__name__]
            answer_options = ["I am " + doing, doing]
        return random.choice(answer_options), None


"""This class represents a sub-type of the DialogueObject to register feedback /
reward given by the user in the form of chat."""


class GetReward(DialogueObject):
    def step(self):
        """associate pos / neg reward to chat memory."""
        self.finished = True
        chatmem = self.memory.get_most_recent_incoming_chat()
        if chatmem.chat_text in [
            "that is wrong",
            "that was wrong",
            "that was completely wrong",
            "not that",
            "that looks horrible",
            "that is not what i asked",
            "that is not what i told you to do",
            "that is not what i asked for" "not what i told you to do",
            "you failed",
            "failure",
            "fail",
            "not what i asked for",
        ]:
            self.memory.tag(chatmem.memid, "neg_reward")
            # should probably tag recent actions too? not just the text of the chat
        elif chatmem.chat_text in [
            "good job",
            "that is really cool",
            "that is awesome",
            "awesome",
            "that is amazing",
            "that looks good",
            "you did well",
            "great",
            "good",
            "nice",
        ]:
            self.memory.tag(chatmem.memid, "pos_reward")
        return "Thanks for letting me know.", None


"""This class represents a sub-type of the DialogueObject to answer questions
about the current location of the bot."""


class BotLocationStatus(DialogueObject):
    def step(self):
        """Extract bot's current location."""
        self.finished = True
        # Get the agent's current location
        agent_pos = pos_to_np(self.agent.get_player().pos)
        agent_coordinates = " , ".join([str(pos) for pos in agent_pos])
        answer_options = [
            "I am here at location : %r" % (agent_coordinates),
            "I am standing at : %r" % (agent_coordinates),
            "I am at : %r" % (agent_coordinates),
        ]
        return random.choice(answer_options), None


"""This class represents a sub-type of the DialogueObject to answer questions
about where the bot is heading."""


class BotMoveStatus(DialogueObject):
    def step(self):
        """Extract bot's target coordinates."""
        self.finished = True
        task = self.memory.task_stack_find_lowest_instance("Move")
        if task is None:
            answer_options = [
                "I am not going anywhere",
                "I am not heading anywhere",
                "I am not off to anywhere",
            ]
        else:
            target_coordinates = " , ".join([str(pos) for pos in task.target])
            answer_options = [
                "I am heading to location : %r" % (target_coordinates),
                "I am walking over to : %r" % (target_coordinates),
                "I am moving to : %r" % (target_coordinates),
            ]
        return random.choice(answer_options), None


"""This class represents a sub-type of the DialogueObject to ask a clarification
question about something."""


class ConfirmTask(DialogueObject):
    def __init__(self, question, tasks, **kwargs):
        super().__init__(**kwargs)
        self.question = question  # chat text that will be sent to user
        self.tasks = tasks  # list of Task objects, will be pushed in order
        self.asked = False

    def step(self):
        """Ask a confirmation question and wait for response."""
        # Step 1: ask the question
        if not self.asked:
            self.dialogue_stack.append_new(AwaitResponse)
            self.dialogue_stack.append_new(Say, self.question)
            self.asked = True
            return "", None

        # Step 2: check the response and add the task if necessary
        self.finished = True
        if len(self.progeny_data) == 0:
            return None, None
        if hasattr(self.progeny_data[-1]["response"], "chat_text"):
            response_str = self.progeny_data.pop(-1)["response"].chat_text
        else:
            response_str = "UNK"
        if response_str in MAP_YES:
            for task in self.tasks:
                self.memory.task_stack_push(task)
        return None, None

class DefineParse(DialogueObject):
    def __init__(self, chat, **kwargs):
        super().__init__(**kwargs)
        self.chat = chat
        self.asked = False
        self.question = "Define {}".format(chat[1])
    
    def step(self):
        if not self.asked:
            self.dialogue_stack.append_new(AwaitResponse, 10000)
            self.dialogue_stack.append_new(Say, self.question)
            self.asked = True
            return "", None

        if len(self.progeny_data) == 0:
            return None, None
        self.finished = True
        if hasattr(self.progeny_data[-1]["response"], "chat_text"):
            decomposition = self.progeny_data.pop(-1)["response"].chat_text
            self.agent.dialogue_manager.online_decomposition(self.chat, decomposition)
            response_str = "Ok, I will execute those commands one at a time now."
        else:
            response_str = "UNK"
        return response_str, None
        
class ConfirmParse(DialogueObject):
    def __init__(self, chat, parse, **kwargs):
        super().__init__(**kwargs)
        self.chat = chat
        self.parse = parse
        # TODO, make human readable
        self.question = "Does this look correct: " + parse['action_type'].__repr__()
        self.asked = False

    def step(self):
        """Ask a confirmation question and wait for response."""
        # Step 1: ask the question
        if not self.asked:
            self.dialogue_stack.append_new(AwaitResponse, 10000)
            self.dialogue_stack.append_new(Say, self.question)
            self.asked = True
            return "", None

        # Step 2: check the response and add the task if necessary
        if len(self.progeny_data) == 0:
            return None, None
        self.finished = True
        if hasattr(self.progeny_data[-1]["response"], "chat_text"):
            response_str = self.progeny_data.pop(-1)["response"].chat_text
        else:
            response_str = "UNK"
        if response_str in MAP_YES:
            output_data = {"response": "yes"}
        elif response_str in MAP_NO:
            self.dialogue_stack.append_new(DefineParse, self.chat)
            output_data = {"response": "no"}
        else:
            output_data = {"response": "unknown"}
        return "", output_data

"""This class represents a sub-type of the DialogueObject to confirm if the
reference object is correct."""

class ConfirmReferenceObject(DialogueObject):
    def __init__(self, reference_object, **kwargs):
        super().__init__(**kwargs)
        r = reference_object
        if hasattr(r, "get_point_at_target"):
            self.bounds = r.get_point_at_target()
        else:
            # this should be an error
            self.bounds = tuple(np.min(r, axis=0)) + tuple(np.max(r, axis=0))
        self.pointed = False
        self.asked = False

    def step(self):
        """Confirm the block object by pointing and wait for answer."""
        if not self.asked:
            self.dialogue_stack.append_new(Say, "do you mean this?")
            self.asked = True
            return "", None
        if not self.pointed:
            self.agent.point_at(self.bounds)
            self.dialogue_stack.append_new(AwaitResponse)
            self.pointed = True
            return "", None
        self.finished = True
        if len(self.progeny_data) == 0:
            output_data = None
        else:
            if hasattr(self.progeny_data[-1]["response"], "chat_text"):
                response_str = self.progeny_data[-1]["response"].chat_text
            else:
                response_str = "UNK"
            if response_str in MAP_YES:
                output_data = {"response": "yes"}
            elif response_str in MAP_NO:
                output_data = {"response": "no"}
            else:
                output_data = {"response": "unkown"}
        return "", output_data

class RequestChange(DialogueObject):
    def __init__(self, obj_name, **kwargs):
        super().__init__(**kwargs)
        self.init_time = self.memory.get_time()
        self.asked = False
        self.obj_name = obj_name
        self.awaiting_response = True
        self.max_steps = 3000000
        f = {"triples": [{"pred_text": "has_tag", "obj_text": "house"}]}
        house_blocks = self.memory.get_reference_objects(f)
        self.house_blocks = house_blocks[0].blocks
        for hb in house_blocks:
            if len(hb.blocks) > len(self.house_blocks):
                self.house_blocks = hb.blocks
        # TODO: too many houses? hackily pick largest house object

    def step(self):
        if not self.asked:
            request_string = "I don't understand what `{}` refers to. "\
                "Please make the change and say 'Done' when you're finished. "\
                "This will help me do better in the future!".format(self.obj_name)
            self.dialogue_stack.append_new(Say, request_string)
            self.asked = True
            return "", None

        chatmem = self.memory.get_most_recent_incoming_chat(after=self.init_time + 1)
        if chatmem is None:
            return "", None
        else:
            import pdb; pdb.set_trace()
            blocks_changed = self.memory.get_player_changed_blocks(self.init_time)[0].blocks
            chat = chatmem.chat_text
            if "done" in chat or "Done" in chat:
                label = self.obj_name
                blocks = [(loc, b) for loc, b in blocks_changed.items()]
                house = [(loc, b) for loc, b in self.house_blocks.items()]
                self.agent.update_segmentation(label, blocks, house)
                output_data = {"response": "done"}
                return "Thanks! I'll remember that in the future.", output_data
            else:
                output_data = {"response": "unknown"}
                return "Sorry, I got confused :(", output_data
 

class AdvancedConfirmReferenceObject(DialogueObject):
    def __init__(self, reference_objects, **kwargs):
        super().__init__(**kwargs)
        r_list = reference_objects
        self.locs_list = [r.blocks for r in r_list]
        self.pointed = False
        self.asked = False

    def step(self):
        if not self.asked:
            request_string = "I don't understand what you're referring to. "\
                "Please point to the object with your cursor and chat `point` "\
                "when you're done pointing."
            self.dialogue_stack.append_new(Say, request_string)
            self.asked = True
            return "", None

        if len(self.progeny_data) == 0:
            return "", None
        else:
            if hasattr(self.progeny_data[-1]["response"], "chat_text"):
                response_str = self.progeny_data.pop(-1)["response"].chat_text
            else:
                response_str = "UNK"
            
            if response_str == "point":
                output_data = {"response": "point"}
                return "got it, thanks!", output_data
            else:
                output_data = {"response": "unknown"}
                return "", output_data
            

        """
        if not self.asked:
            request_string = "I don't know what you are referring to. Can you "\
                "help me locate it using one of the objects I know? Here is "\
                "what I see."
            self.dialogue_stack.append_new(Say, request_string)
            self.asked = True
            return "", None
        if not self.pointed:
            for i, locs in enumerate(self.locs_list):
                self.agent.point_s_at(locs)
                self.agent.send_chat("Candidate {}".format(i))
            self.pointed = True
            self.dialogue_stack.append_new(AwaitResponse)
            return "", None
        self.finished = True
        if len(self.progeny_data) == 0:
            output_data = None
        else:
            if hasattr(self.progeny_data[-1]["response"], "chat_text"):
                response_str = self.progeny_data[-1]["response"].chat_text
            else:
                response_str = "UNK"
            if response_str in MAP_YES:
                output_data = {"response": "yes"}
            elif response_str in MAP_NO:
                output_data = {"response": "no"}
            else:
                output_data = {"response": "unkown"}
        return "", output_data
        """
