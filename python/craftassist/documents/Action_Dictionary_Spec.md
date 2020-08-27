Note: Classes used in the dict using `<>` are defined as subsections and their corresponding dicts should be substituted in place.

# Dialogue Types #

## Human Give Command Dialogue type ##
```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : list(<CommandDict>)
}
```
The CommandDict for each action type is described in the [Action subsection]().

## Noop Dialogue Type ##
```
{ "dialogue_type": "NOOP"}
```

## Get Memory Dialogue Type ##
```
{
  "dialogue_type": "GET_MEMORY",
  "filters": {
    "temporal": CURRENT,
    "type": "ACTION" / "AGENT" / "REFERENCE_OBJECT",
    "action_type": BUILD / DESTROY / DIG / FILL / SPAWN / MOVE
    "reference_object" : {
      <Location>,
      "has_size" : span,
      "has_colour" : span,
      "has_name" : span,
      "coref_resolve": span,
      },
    },
  "answer_type": "TAG" / "EXISTS" ,
  "tag_name" : 'has_name' / 'has_size' / 'has_colour' / 'action_name' /
              'action_reference_object_name' / 'move_target' / 'location' ,
  "replace": true
}
```
## Put Memory Dialogue Type ##
```
{
  "dialogue_type": "PUT_MEMORY",
  "filters": {
    "reference_object" : {
      <Location>,
      "has_size" : span,
      "has_colour" : span,
      "has_name" : span,
      "coref_resolve" : span,
      <Repeat>
     },
  },
  "upsert" : {
      "memory_data": {
        "memory_type": "REWARD" / "TRIPLE",
        "reward_value": "POSITIVE" / "NEGATIVE",
        "has_tag" : span,
        "has_colour": span,
        "has_size": span
      } }
}
```

## Actions ##

### Build Action ###
This is the action to Build a schematic at an optional location.

```
{ "action_type" : BUILD,
  <Location>,
  <Schematic>,
  <Repeat> (with repeat_key: 'FOR' and additional repeat_dir: 'SURROUND'),
  "replace" : True
}
    
```
### Copy Action ###
This is the action to copy a block object to an optional location. The copy action is represented as a "Build" with an optional reference_object in the tree.

```
{ "action_type" : 'BUILD',
  <Location>,
  <ReferenceObject>,
  <Repeat> (repeat_key = 'FOR'),
  "replace" : True
}
```

### Spawn Action ###
This action indicates that the specified object should be spawned in the environment.
Spawn only has a name in the reference object.

```
{ "action_type" : 'SPAWN',
  "reference_object" : {
      "text_span" : span,
      <Repeat>(repeat_key= 'FOR'),
      "has_name" : span,
    },
    <Repeat>(repeat_key= 'FOR'),
    "replace": True
}
```

### Resume ###
This action indicates that the previous action should be resumed.

```
{ "action_type" : 'RESUME',
  "target_action_type": span
}
```

### Fill ###
This action states that a hole / negative shape needs to be filled up.

```
{ "action_type" : 'FILL',
  "has_block_type" : span,
  <ReferenceObject>,
  <Repeat>,
  "replace": True
}
```

#### Destroy ####
This action indicates the intent to destroy a block object.

```
{ "action_type" : 'DESTROY',
  <ReferenceObject>,
  <Repeat>,
  "replace": True
}
```

#### Move ####
This action states that the agent should move to the specified location.

```
{ "action_type" : 'MOVE',
  <Location>,
  <StopCondition>,
  <Repeat>,
  "replace": True
}
```

#### Undo ####
This action states the intent to revert the specified action, if any.

```
{ "action_type" : 'UNDO',
  "target_action_type" : span
}
```

#### Stop ####
This action indicates stop.

```
{ "action_type" : 'STOP',
  "target_action_type": span
}
```

#### Dig ####
This action represents the intent to dig a hole / negative shape of optional dimensions at an optional location.
The `Schematic` child in this only has a subset of properties.

```
{ "action_type" : 'DIG',
  <Location>,
  "schematic" : {
    "text_span" : span,
    <Repeat>(repeat_key = 'FOR'),
     "has_size" : span,
     "has_length" : span,
     "has_depth" : span,
     "has_width" : span
     },
  <StopCondition>,
  <Repeat>,
  "replace": True  
}
```

#### FreeBuild ####
This action represents that the agent should complete an already existing half-finished block object, using its mental model.

```
{ "action_type" : 'FREEBUILD',
  <ReferenceObject>,
  <Location>,
  <Repeat>,
  "replace": True
}
```

#### Dance ####
This action provides information to the agent to do a dance.
Also has support for Point / Turn / Look.

```
{ "action_type" : 'DANCE',
  <Location> (additional relative_direction values: ['CLOCKWISE', 'ANTICLOCKWISE']),
  <DanceType>
  "stop_condition" : {
      "condition_type" : NEVER,
  },
  "repeat" : {
    "repeat_key" : 'FOR',
    "repeat_count" : span, # Note no repeat_dir here.
  },
  "replace": True
}
```

#### Get ####
The GET action_type covers the intents: bring, get and give.

The Get intent represents getting or picking up something. This might involve first going to that thing and then picking it up in bot’s hand. The receiver here can either be the agent or the speaker (or another player).
The Give intent represents giving something, in Minecraft this would mean removing something from the inventory of the bot and adding it to the inventory of the speaker / other player.
The Bring intent represents bringing a reference_object to the speaker or to a specified location.

```
{
    "action_type" : 'GET',
    <ReferenceObject>,
    <Repeat>,
    "receiver" : <ReferenceObject> / <Location>
}
```

#### Scout ####
This command expresses the intent to look for / find or scout something.
```
{
    "action_type" : 'SCOUT',
    <ReferenceObject>
}
```

#### Modify action ####
Note that: This section is a temporary heristic based fix until we have a generative model that can handle "modify".

This command represents making a change or modifying a block object in a certain way.

[Examples](https://docs.google.com/document/d/1nLEMRvUO9VNV_HYVRB7c9kDkUl0dACzwjY0h_APDcVU/edit?ts=5e050a45#bookmark=id.o3lwuh3fj1lt)

Grammar proposal:
```
{ "dialogue_type": "HUMAN_GIVE_COMMAND",
  "action_sequence" : [
    { "action_type" : 'MODIFY',
      "reference_object" : REFERENCE_OBJECT,
      "modify_dict" :  THICKEN/SCALE/RIGIDMOTION/REPLACE/FILL
    }]
}
```
Where
```
THICKEN = {"modify_type": "THICKER"/"THINNER", "num_blocks": span}
SCALE = {"modify_type": "SCALE", "text_scale_factor": span, "categorical_scale_factor": "WIDER"/"NARROWER"/"TALLER"/"SHORTER"/"SKINNIER"/"FATTER"/"BIGGER"/"SMALLER"}
RIGIDMOTION = {"modify_type": "RIGIDMOTION", "categorical_angle": "LEFT"/"RIGHT"/"AROUND", "MIRROR"=True/False, "location": LOCATION}
REPLACE={"modify_type": "REPLACE", "old_block": BLOCK, "new_block": BLOCK, "replace_geometry": REPLACE_GEOMETRY}
FILL={"modify_type": "FILL"/"HOLLOW", "new_block": BLOCK}
```
And
```
BLOCK = {"has_x": span}
REPLACE_GEOMETRY = {"relative_direction": "LEFT"/"RIGHT"/"TOP"/"BOTTOM", "amount": "QUARTER"/"HALF"}
```
Note w2n doesn't handle fractions :(

If "location" is given in RIGIDMOTION, the modify is to move the reference object from its current location to the given location


### Subcomponents of action dict ###

#### Location ####
```
"location" : {
          "text_span" : span,
          "steps" : span,
          "has_measure" : span,
          "contains_coreference" : "yes",
          "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/ 'AWAY'
                                  / 'INSIDE' / 'NEAR' / 'OUTSIDE' / 'BETWEEN',
          <ReferenceObject>,
          }
 ```

Note: for "relative_direction" == 'BETWEEN' the location dict will have two children: 'reference_object_1' : <ReferenceObject>['reference_object'] and 
'reference_object_2' : <ReferenceObject>['reference_object'] in the sub-dictionary to represent the two distinct reference objects.

#### Reference Object ####
```
"reference_object" : {
      "text_span" : span,
      <Repeat>,
      "special_reference" : 'SPEAKER' / 'AGENT' / 'SPEAKER_LOOK' / {'coordinates_span' : span},
      "filters" : {
              "has_name" : span,
              "has_size" : span,
              "has_colour" : span,
              "contains_coreference" : "yes",
              "location" : {
                  "contains_coreference" : "yes",
                  "steps" : span,
                  "relative_direction" : 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK'/ 'AWAY' / 'NEAR',
                  <ReferenceObject> (with only: "special_reference", "has_name", "has_size", "has_color" and "contains_coreference" fields)
               } 
      }
  } 
```
#### Stop Condition ####
```
"stop_condition" : {
      "condition_type" : 'ADJACENT_TO_BLOCK_TYPE' / 'NEVER',
      "block_type" : span
  }
```
#### Schematic ####

```
"schematic" : {
          "text_span" : span,
          <Repeat> (with repeat_key: 'FOR' and additional 'SURROUND' repeat_dir), 
          "has_block_type" : span,
          "has_name": span,
          "has_size" : span,
          "has_orientation" : span,
          "has_thickness" : span,
          "has_colour" : span,
          "has_height" : span,
          "has_length" : span,
          "has_radius" : span,
          "has_slope" : span,
          "has_width" : span,
          "has_base" : span,
          "has_depth" : span,
          "has_distance" : span,
      }
```
#### Repeat ####
```
"repeat" : {
            "repeat_key" : 'FOR'/ 'ALL'
            "repeat_count" : span,
            "repeat_dir": 'LEFT' / 'RIGHT'/ 'UP'/ 'DOWN'/ 'FRONT'/ 'BACK' / 'AROUND'
      }
```

#### FACING ####
```
{
  "text_span" : span,
  "yaw_pitch": span,
  "yaw": span,
  "pitch": span,
  "relative_yaw" = {"angle": -360, -180, -135, -90, -45, 45, 90, 135, 180, 360 
  		    "yaw_span": span},
   "relative_pitch" = {"angle": -90, -45, 45, 90, 
  		    "pitch_span": span},
  "location": <LOCATION>
}
```

#### DanceType ####
```
"dance_type" : {
  "dance_type_name": span,
  "dance_type_tag": span,
  "point": <FACING>,
  "look_turn": <FACING>,
  "body_turn": <FACING>
}
```

