from base_agent.dialogue_objects import SPEAKERLOOK

INTERPRETER_POSSIBLE_ACTIONS = {
    "destroy_speaker_look": {
        "action_type": "DESTROY",
        "reference_object": {
            "filters": {"location": SPEAKERLOOK},
            "text_span": "where I'm looking",
        },
    },
    "spawn_5_sheep": {
        "action_type": "SPAWN",
        "reference_object": {"filters": {"has_name": "sheep"}, "text_span": "sheep"},
        "repeat": {"repeat_key": "FOR", "repeat_count": "5"},
    },
    "copy_speaker_look_to_agent_pos": {
        "action_type": "BUILD",
        "reference_object": {
            "filters": {"location": SPEAKERLOOK},
            "text_span": "where I'm looking",
        },
        "location": {
            "reference_object": {"special_reference": "AGENT"},
            "text_span": "where I am",
        },
    },
    "build_small_sphere": {
        "action_type": "BUILD",
        "schematic": {"has_name": "sphere", "has_size": "small", "text_span": "small sphere"},
    },
    "build_1x1x1_cube": {
        "action_type": "BUILD",
        "schematic": {"has_name": "cube", "has_size": "1 x 1 x 1", "text_span": "1 x 1 x 1 cube"},
    },
    "move_speaker_pos": {
        "action_type": "MOVE",
        "location": {"reference_object": {"special_reference": "SPEAKER"}, "text_span": "to me"},
    },
    "build_diamond": {
        "action_type": "BUILD",
        "schematic": {"has_name": "diamond", "text_span": "diamond"},
    },
    "build_gold_cube": {
        "action_type": "BUILD",
        "schematic": {"has_block_type": "gold", "has_name": "cube", "text_span": "gold cube"},
    },
    "build_red_cube": {
        "action_type": "BUILD",
        "location": {"reference_object": {"special_reference": "SPEAKER_LOOK"}},
        "schematic": {"has_colour": "red", "has_name": "cube", "text_span": "red cube"},
    },
    "destroy_red_cube": {
        "action_type": "DESTROY",
        "reference_object": {
            "filters": {"has_name": "cube", "has_colour": "red"},
            "text_span": "red cube",
        },
    },
    "fill_all_holes_speaker_look": {
        "action_type": "FILL",
        "reference_object": {
            "filters": {"location": SPEAKERLOOK},
            "text_span": "where I'm looking",
        },
        "repeat": {"repeat_key": "ALL"},
    },
    "go_to_tree": {
        "action_type": "MOVE",
        "location": {"reference_object": {"filters": {"has_name": "tree"}}, "text_span": "tree"},
    },
    "build_square_height_1": {
        "action_type": "BUILD",
        "schematic": {"has_name": "square", "has_height": "1", "text_span": "square height 1"},
    },
    "stop": {"action_type": "STOP"},
    "fill_speaker_look": {
        "action_type": "FILL",
        "reference_object": {
            "filters": {"location": SPEAKERLOOK},
            "text_span": "where I'm looking",
        },
    },
    "fill_speaker_look_gold": {
        "action_type": "FILL",
        "has_block_type": "gold",
        "reference_object": {
            "filters": {"location": SPEAKERLOOK},
            "text_span": "where I'm looking",
        },
    },
}

BUILD_COMMANDS = {
    "build a gold cube at 0 66 0": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "BUILD",
                "schematic": {"has_name": "cube", "has_block_type": "gold"},
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "0 66 0"}}
                },
            }
        ],
    },
    "build a small cube": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {"action_type": "BUILD", "schematic": {"has_name": "cube", "has_size": "small"}}
        ],
    },
    "build a circle to the left of the circle": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "BUILD",
                "location": {
                    "reference_object": {"filters": {"has_name": "circle"}},
                    "relative_direction": "LEFT",
                    "text_span": "to the left of the circle",
                },
                "schematic": {"has_name": "circle", "text_span": "circle"},
            }
        ],
    },
    "copy where I am looking to here": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["copy_speaker_look_to_agent_pos"]],
    },
    "build a small sphere": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["build_small_sphere"]],
    },
    "build a 1x1x1 cube": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["build_1x1x1_cube"]],
    },
    "build a diamond": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["build_diamond"]],
    },
    "build a gold cube": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["build_gold_cube"]],
    },
    "build a 9 x 9 stone rectangle": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "BUILD",
                "schematic": {
                    "has_block_type": "stone",
                    "has_name": "rectangle",
                    "has_height": "9",
                    "has_base": "9",  # has_base doesn't belong in "rectangle"
                    "text_span": "9 x 9 stone rectangle",
                },
            }
        ],
    },
    "build a square with height 1": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["build_square_height_1"]],
    },
    "build a red cube": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["build_red_cube"]],
    },
    "build a fluffy here": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "BUILD",
                "schematic": {"has_name": "fluffy"},
                "location": {"reference_object": {"special_reference": "AGENT"}},
            }
        ],
    },
}

SPAWN_COMMANDS = {
    "spawn 5 sheep": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["spawn_5_sheep"]],
    }
}

DESTROY_COMMANDS = {
    "destroy it": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "DESTROY",
                "reference_object": {"filters": {"contains_coreference": "yes"}},
            }
        ],
    },
    "destroy where I am looking": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["destroy_speaker_look"]],
    },
    "destroy the red cube": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["destroy_red_cube"]],
    },
    "destroy everything": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "reference_object": {
                    "repeat": {"repeat_key": "ALL"},
                    "filters": {},
                    "text_span": "everything",
                },
                "action_type": "DESTROY",
            }
        ],
    },
    "destroy the fluff thing": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {"action_type": "DESTROY", "reference_object": {"filters": {"has_tag": "fluff"}}}
        ],
    },
    "destroy the fluffy object": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {"action_type": "DESTROY", "reference_object": {"filters": {"has_tag": "fluffy"}}}
        ],
    },
}

MOVE_COMMANDS = {
    "move to 42 65 0": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "42 65 0"}}
                },
            }
        ],
    },
    "move to 0 63 0": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "0 63 0"}}
                },
            }
        ],
    },
    "move to -7 63 -8": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "-7 63 -8"}},
                    "text_span": "-7 63 -8",
                },
            }
        ],
    },
    "go between the cubes": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"filters": {"has_name": "cube"}},
                    "relative_direction": "BETWEEN",
                    "text_span": "between the cubes",
                },
            }
        ],
    },
    "move here": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["move_speaker_pos"]],
    },
    "go to the tree": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["go_to_tree"]],
    },
    "move to 20 63 20": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "20 63 20"}},
                    "text_span": "20 63 20",
                },
            }
        ],
    },
}

FILL_COMMANDS = {
    "fill where I am looking": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["fill_speaker_look"]],
    },
    "fill where I am looking with gold": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["fill_speaker_look_gold"]],
    },
    "fill all holes where I am looking": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["fill_all_holes_speaker_look"]],
    },
}

DANCE_COMMANDS = {
    "dance": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [{"action_type": "DANCE", "dance_type": {"dance_type_span": "dance"}}],
    }
}

COMBINED_COMMANDS = {
    "build a small sphere then move here": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            INTERPRETER_POSSIBLE_ACTIONS["build_small_sphere"],
            INTERPRETER_POSSIBLE_ACTIONS["move_speaker_pos"],
        ],
    },
    "copy where I am looking to here then build a 1x1x1 cube": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            INTERPRETER_POSSIBLE_ACTIONS["copy_speaker_look_to_agent_pos"],
            INTERPRETER_POSSIBLE_ACTIONS["build_1x1x1_cube"],
        ],
    },
    "move to 3 63 2 then 7 63 7": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "3 63 2"}},
                    "text_span": "3 63 2",
                },
            },
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": {"coordinates_span": "7 63 7"}},
                    "text_span": "7 63 7",
                },
            },
        ],
    },
}

GET_MEMORY_COMMANDS = {
    "what is where I am looking": {
        "dialogue_type": "GET_MEMORY",
        "filters": {"type": "REFERENCE_OBJECT", "reference_object": {"location": SPEAKERLOOK}},
        "answer_type": "TAG",
        "tag_name": "has_name",
    },
    "what are you doing": {
        "dialogue_type": "GET_MEMORY",
        "filters": {"type": "ACTION"},
        "answer_type": "TAG",
        "tag_name": "action_name",
    },
    "what are you building": {
        "dialogue_type": "GET_MEMORY",
        "filters": {"type": "ACTION", "action_type": "BUILD"},
        "answer_type": "TAG",
        "tag_name": "action_reference_object_name",
    },
    "where are you going": {
        "dialogue_type": "GET_MEMORY",
        "filters": {"type": "ACTION", "action_type": "MOVE"},
        "answer_type": "TAG",
        "tag_name": "move_target",
    },
    "where are you": {
        "dialogue_type": "GET_MEMORY",
        "filters": {"type": "AGENT"},
        "answer_type": "TAG",
        "tag_name": "location",
    },
}

PUT_MEMORY_COMMANDS = {
    "that is fluff": {
        "dialogue_type": "PUT_MEMORY",
        "filters": {"reference_object": {"location": SPEAKERLOOK}},
        "upsert": {"memory_data": {"memory_type": "TRIPLE", "has_tag": "fluff"}},
    },
    "good job": {
        "dialogue_type": "PUT_MEMORY",
        "upsert": {"memory_data": {"memory_type": "REWARD", "reward_value": "POSITIVE"}},
    },
    "that is fluffy": {
        "dialogue_type": "PUT_MEMORY",
        "filters": {"reference_object": {"location": SPEAKERLOOK}},
        "upsert": {"memory_data": {"memory_type": "TRIPLE", "has_tag": "fluffy"}},
    },
}

OTHER_COMMANDS = {
    "the weather is good": {"dialogue_type": "NOOP"},
    "stop": {
        "dialogue_type": "HUMAN_GIVE_COMMAND",
        "action_sequence": [INTERPRETER_POSSIBLE_ACTIONS["stop"]],
    },
    "undo": {"dialogue_type": "HUMAN_GIVE_COMMAND", "action_sequence": [{"action_type": "UNDO"}]},
}

GROUND_TRUTH_PARSES = {
    "go to the gray chair": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"filters": {"has_colour": "gray", "has_name": "chair"}}
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go to the chair": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {"reference_object": {"filters": {"has_name": "chair"}}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go forward 0.2 meters": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "FRONT",
                    "steps": "0.2",
                    "has_measure": "meters",
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go forward one meter": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "FRONT",
                    "steps": "one",
                    "has_measure": "meter",
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go left 3 feet": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "LEFT",
                    "steps": "3",
                    "has_measure": "feet",
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go left 3 meters": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "LEFT",
                    "steps": "3",
                    "has_measure": "meters",
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go forward 1 feet": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "FRONT",
                    "steps": "1",
                    "has_measure": "feet",
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go back 1 feet": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {
                    "reference_object": {"special_reference": "AGENT"},
                    "relative_direction": "BACK",
                    "steps": "1",
                    "has_measure": "feet",
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "turn right 90 degrees": {
        "action_sequence": [
            {
                "action_type": "DANCE",
                "dance_type": {"body_turn": {"relative_yaw": {"angle": "90"}}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "turn left 90 degrees": {
        "action_sequence": [
            {
                "action_type": "DANCE",
                "dance_type": {"body_turn": {"relative_yaw": {"angle": "-90"}}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "turn right 180 degrees": {
        "action_sequence": [
            {
                "action_type": "DANCE",
                "dance_type": {"body_turn": {"relative_yaw": {"angle": "180"}}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "turn right": {
        "action_sequence": [
            {
                "action_type": "DANCE",
                "dance_type": {"body_turn": {"relative_yaw": {"angle": "90"}}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "look at where I am pointing": {
        "action_sequence": [
            {
                "action_type": "DANCE",
                "dance_type": {
                    "look_turn": {
                        "location": {"reference_object": {"special_reference": "SPEAKER_LOOK"}}
                    }
                },
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "wave": {
        "action_sequence": [{"action_type": "DANCE", "dance_type": {"dance_type_name": "wave"}}],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "follow the chair": {
        "action_sequence": [
            {
                "action_type": "MOVE",
                "location": {"reference_object": {"filters": {"has_name": "chair"}}},
                "stop_condition": {"condition_type": "NEVER"},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "find Laurens": {
        "action_sequence": [
            {"action_type": "SCOUT", "reference_object": {"filters": {"has_name": "Laurens"}}}
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "bring the cup to Mary": {
        "action_sequence": [
            {
                "action_type": "GET",
                "receiver": {"reference_object": {"filters": {"has_name": "Mary"}}},
                "reference_object": {"filters": {"has_name": "cup"}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
    "go get me lunch": {
        "action_sequence": [
            {
                "action_type": "GET",
                "receiver": {"reference_object": {"special_reference": "SPEAKER"}},
                "reference_object": {"filters": {"has_name": "lunch"}},
            }
        ],
        "dialogue_type": "HUMAN_GIVE_COMMAND",
    },
}
