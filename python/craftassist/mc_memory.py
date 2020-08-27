"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import random
import sys
from typing import Optional, List

from build_utils import npy_to_blocks_list
import minecraft_specs
import dance

PERCEPTION_RANGE = 64
BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(BASE_AGENT_ROOT)


from base_agent.util import XYZ, Block

from base_agent.sql_memory import AgentMemory

from base_agent.memory_nodes import (  # noqa
    TaskNode,
    PlayerNode,
    MemoryNode,
    ChatNode,
    TimeNode,
    LocationNode,
    SetNode,
    ReferenceObjectNode,
)

from mc_memory_nodes import (  # noqa
    DanceNode,
    VoxelObjectNode,
    BlockObjectNode,
    BlockTypeNode,
    MobNode,
    ItemStackNode,
    MobTypeNode,
    InstSegNode,
    SchematicNode,
    NODELIST,
)

from word_maps import SPAWN_OBJECTS

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..")

SCHEMAS = [
    os.path.join(os.path.join(BASE_AGENT_ROOT, "base_agent"), "base_memory_schema.sql"),
    os.path.join(os.path.dirname(__file__), "mc_memory_schema.sql"),
]

SCHEMA = os.path.join(os.path.dirname(__file__), "memory_schema.sql")

THROTTLING_TICK_UPPER_LIMIT = 64
THROTTLING_TICK_LOWER_LIMIT = 4

# TODO "snapshot" memory type  (giving a what mob/object/player looked like at a fixed timestamp)
# TODO when a memory is removed, its last state should be snapshotted to prevent tag weirdness


class MCAgentMemory(AgentMemory):
    def __init__(
        self,
        db_file=":memory:",
        db_log_path=None,
        schema_paths=SCHEMAS,
        load_minecraft_specs=True,
        load_block_types=True,
        load_mob_types=True,
        preception_range=PERCEPTION_RANGE,
    ):
        super(MCAgentMemory, self).__init__(
            db_file=db_file, schema_paths=schema_paths, db_log_path=db_log_path, nodelist=NODELIST
        )
        self.banned_default_behaviors = []  # FIXME: move into triple store?
        self._safe_pickle_saved_attrs = {}
        self._load_schematics(load_minecraft_specs)
        self._load_block_types(load_block_types)
        self._load_mob_types(load_mob_types)

        self.dances = {}
        dance.add_default_dances(self)

        self.perception_range = preception_range

    ########################
    ### ReferenceObjects ###
    ########################

    def get_entity_by_eid(self, eid) -> Optional["ReferenceObjectNode"]:
        r = self._db_read_one("SELECT uuid FROM ReferenceObjects WHERE eid=?", eid)
        if r:
            return self.get_mem_by_id(r[0])
        else:
            return None

    ###############
    ### Voxels  ###
    ###############

    # count updates are done by hand to not need to count all voxels every time
    # use these functions, don't add/delete/modify voxels with raw sql

    def update_voxel_count(self, memid, dn):
        c = self._db_read_one("SELECT voxel_count FROM ReferenceObjects WHERE uuid=?", memid)
        if c:
            count = c[0] + dn
            self._db_write("UPDATE ReferenceObjects SET voxel_count=? WHERE uuid=?", count, memid)
            return count
        else:
            return None

    def update_voxel_mean(self, memid, count, loc):
        """ update the x, y, z entries in ReferenceObjects
        to account for the removal or addition of a block.
        count should be the number of voxels *after* addition if >0
        and -count the number *after* removal if count < 0
        count should not be 0- handle that outside
        """
        old_loc = self._db_read_one("SELECT x, y, z  FROM ReferenceObjects WHERE uuid=?", memid)
        # TODO warn/error if no such memory?
        assert count != 0
        if old_loc:
            b = 1 / count
            if count > 0:
                a = (count - 1) / count
            else:
                a = (1 - count) / (-count)
            new_loc = (
                old_loc[0] * a + loc[0] * b,
                old_loc[1] * a + loc[1] * b,
                old_loc[2] * a + loc[2] * b,
            )
            self._db_write(
                "UPDATE ReferenceObjects SET x=?, y=?, z=? WHERE uuid=?", *new_loc, memid
            )
            return new_loc

    def remove_voxel(self, x, y, z, ref_type):
        memids = self._db_read_one(
            "SELECT uuid FROM VoxelObjects WHERE x=? and y=? and z=? and ref_type=?",
            x,
            y,
            z,
            ref_type,
        )
        if not memids:
            # TODO error/warning?
            return
        memid = memids[0]
        c = self.update_voxel_count(memid, -1)
        if c > 0:
            self.update_voxel_mean(memid, c, (x, y, z))
        self._db_write(
            "DELETE FROM VoxelObjects WHERE x=? AND y=? AND z=? and ref_type=?", x, y, z, ref_type
        )
        if c == 0:
            #        if not self.memory.check_memid_exists(memid, "VoxelObjects"):
            # object is gone now.  TODO be more careful here... maybe want to keep some records?
            self.remove_memid_triple(memid, role="both")

    # this only upserts to the same ref_type- if the voxel is occupied by
    # a different ref_type it will insert a new ref object even if update is True
    def upsert_block(
        self,
        block: Block,
        memid: str,
        ref_type: str,
        player_placed: bool = False,
        agent_placed: bool = False,
        update: bool = True,  # if update is set to False, forces a write
    ):
        ((x, y, z), (b, m)) = block
        old_memid = self._db_read_one(
            "SELECT uuid FROM VoxelObjects WHERE x=? AND y=? AND z=? and ref_type=?",
            x,
            y,
            z,
            ref_type,
        )
        # add to voxel count
        new_count = self.update_voxel_count(memid, 1)
        assert new_count
        self.update_voxel_mean(memid, new_count, (x, y, z))
        if old_memid and update:
            if old_memid != memid:
                self.remove_voxel(x, y, z, ref_type)
                cmd = "INSERT INTO VoxelObjects (uuid, bid, meta, updated, player_placed, agent_placed, ref_type, x, y, z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            else:
                cmd = "UPDATE VoxelObjects SET uuid=?, bid=?, meta=?, updated=?, player_placed=?, agent_placed=? WHERE ref_type=? AND x=? AND y=? AND z=?"
        else:
            cmd = "INSERT INTO VoxelObjects (uuid, bid, meta, updated, player_placed, agent_placed, ref_type, x, y, z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        self._db_write(
            cmd, memid, b, m, self.get_time(), player_placed, agent_placed, ref_type, x, y, z
        )

    ######################
    ###  BlockObjects  ###
    ######################

    # rename this... "object" is bad name
    def get_object_by_id(self, memid: str, table="BlockObjects") -> "VoxelObjectNode":
        if table == "BlockObjects":
            return BlockObjectNode(self, memid)
        elif table == "InstSeg":
            return InstSegNode(self, memid)
        else:
            raise ValueError("Bad table={}".format(table))

    # and rename this
    def get_object_info_by_xyz(self, xyz: XYZ, ref_type: str, just_memid=True):
        r = self._db_read(
            "SELECT DISTINCT(uuid), bid, meta FROM VoxelObjects WHERE x=? AND y=? AND z=? and ref_type=?",
            *xyz,
            ref_type
        )
        if just_memid:
            return [memid for (memid, bid, meta) in r]
        else:
            return r

    # WARNING: these do not search archived/snapshotted block objects
    # TODO replace all these all through the codebase with generic counterparts
    def get_block_object_ids_by_xyz(self, xyz: XYZ) -> List[str]:
        return self.get_object_info_by_xyz(xyz, "BlockObjects")

    def get_block_object_by_xyz(self, xyz: XYZ) -> Optional["VoxelObjectNode"]:
        memids = self.get_block_object_ids_by_xyz(xyz)
        if len(memids) == 0:
            return None
        return self.get_block_object_by_id(memids[0])

    def get_block_object_by_id(self, memid: str) -> "VoxelObjectNode":
        return self.get_object_by_id(memid, "BlockObjects")

    def tag_block_object_from_schematic(self, block_object_memid: str, schematic_memid: str):
        self.add_triple(subj=block_object_memid, pred_text="_from_schematic", obj=schematic_memid)

    #####################
    ### InstSegObject ###
    #####################

    def get_instseg_object_ids_by_xyz(self, xyz: XYZ) -> List[str]:
        r = self._db_read(
            'SELECT DISTINCT(uuid) FROM VoxelObjects WHERE ref_type="inst_seg" AND x=? AND y=? AND z=?',
            *xyz
        )
        return r

    ####################
    ###  Schematics  ###
    ####################

    def get_schematic_by_id(self, memid: str) -> "SchematicNode":
        return SchematicNode(self, memid)

    def get_schematic_by_property_name(self, name, table_name) -> Optional["SchematicNode"]:
        r = self._db_read(
            """
                    SELECT {}.type_name
                    FROM {} INNER JOIN Triples as T ON T.subj={}.uuid
                    WHERE (T.pred_text="has_name" OR T.pred_text="has_tag") AND T.obj_text=?""".format(
                table_name, table_name, table_name
            ),
            name,
        )
        if not r:
            return None

        result = []  # noqa
        for e in r:
            schematic_name = e[0]
            schematics = self._db_read(
                """
                    SELECT Schematics.uuid
                    FROM Schematics INNER JOIN Triples as T ON T.subj=Schematics.uuid
                    WHERE (T.pred_text="has_name" OR T.pred_text="has_tag") AND T.obj_text=?""",
                schematic_name,
            )
            if schematics:
                result.extend(schematics)
        if result:
            return self.get_schematic_by_id(random.choice(result)[0])
        else:
            return None

    def get_mob_schematic_by_name(self, name: str) -> Optional["SchematicNode"]:
        return self.get_schematic_by_property_name(name, "MobTypes")

    # TODO call this in get_schematic_by_property_name
    def get_schematic_by_name(self, name: str) -> Optional["SchematicNode"]:
        r = self._db_read(
            """
                SELECT Schematics.uuid
                FROM Schematics INNER JOIN Triples as T ON T.subj=Schematics.uuid
                WHERE (T.pred_text="has_name" OR T.pred_text="has_tag") AND T.obj_text=?""",
            name,
        )
        if r:  # if multiple exist, then randomly select one
            return self.get_schematic_by_id(random.choice(r)[0])
        # if no schematic with exact matched name exists, search for a schematic
        # with matched property name instead
        else:
            return self.get_schematic_by_property_name(name, "BlockTypes")

    def convert_block_object_to_schematic(self, block_object_memid: str) -> "SchematicNode":
        r = self._db_read_one(
            'SELECT subj FROM Triples WHERE pred_text="_source_block_object" AND obj=?',
            block_object_memid,
        )
        if r:
            # previously converted; return old schematic
            return self.get_schematic_by_id(r[0])

        else:
            # get up to date BlockObject
            block_object = self.get_block_object_by_id(block_object_memid)

            # create schematic
            memid = SchematicNode.create(self, list(block_object.blocks.items()))

            # add triple linking the object to the schematic
            self.add_triple(subj=memid, pred_text="_source_block_object", obj=block_object.memid)

            return self.get_schematic_by_id(memid)

    def _load_schematics(self, load_minecraft_specs=True):
        if load_minecraft_specs:
            for premem in minecraft_specs.get_schematics():
                npy = premem["schematic"]
                memid = SchematicNode.create(self, npy_to_blocks_list(npy))
                if premem.get("name"):
                    for n in premem["name"]:
                        self.add_triple(subj=memid, pred_text="has_name", obj_text=n)
                        self.add_triple(subj=memid, pred_text="has_tag", obj_text=n)
                if premem.get("tags"):
                    for t in premem["tags"]:
                        self.add_triple(subj=memid, pred_text="has_tag", obj_text=t)

        # load single blocks as schematics
        bid_to_name = minecraft_specs.get_block_data()["bid_to_name"]
        for (d, m), name in bid_to_name.items():
            if d >= 256:
                continue
            memid = SchematicNode.create(self, [((0, 0, 0), (d, m))])
            self.add_triple(subj=memid, pred_text="has_name", obj_text=name)
            if "block" in name:
                self.add_triple(
                    subj=memid, pred_text="has_name", obj_text=name.strip("block").strip()
                )
            # tag single blocks with 'block'
            self.add_triple(subj=memid, pred_text="has_name", obj_text="block")

    def _load_block_types(
        self,
        load_block_types=True,
        load_color=True,
        load_block_property=True,
        simple_color=False,
        load_material=True,
    ):
        if not load_block_types:
            return
        bid_to_name = minecraft_specs.get_block_data()["bid_to_name"]

        color_data = minecraft_specs.get_colour_data()
        if simple_color:
            name_to_colors = color_data["name_to_simple_colors"]
        else:
            name_to_colors = color_data["name_to_colors"]

        block_property_data = minecraft_specs.get_block_property_data()
        block_name_to_properties = block_property_data["name_to_properties"]

        for (b, m), type_name in bid_to_name.items():
            if b >= 256:
                continue
            memid = BlockTypeNode.create(self, type_name, (b, m))
            self.add_triple(subj=memid, pred_text="has_name", obj_text=type_name)
            if "block" in type_name:
                self.add_triple(
                    subj=memid, pred_text="has_name", obj_text=type_name.strip("block").strip()
                )

            if load_color:
                if name_to_colors.get(type_name) is not None:
                    for color in name_to_colors[type_name]:
                        self.add_triple(subj=memid, pred_text="has_colour", obj_text=color)

            if load_block_property:
                if block_name_to_properties.get(type_name) is not None:
                    for property in block_name_to_properties[type_name]:
                        self.add_triple(subj_text=memid, pred_text="has_name", obj_text=property)

    def _load_mob_types(self, load_mob_types=True):
        if not load_mob_types:
            return

        mob_property_data = minecraft_specs.get_mob_property_data()
        mob_name_to_properties = mob_property_data["name_to_properties"]
        for (name, m) in SPAWN_OBJECTS.items():
            type_name = "spawn " + name

            # load single mob as schematics
            memid = SchematicNode.create(self, [((0, 0, 0), (383, m))])
            self.add_triple(subj=memid, pred_text="has_name", obj_text=type_name)
            if "block" in type_name:
                self.add_triple(
                    subj=memid, pred_text="has_name", obj_text=type_name.strip("block").strip()
                )

            # then load properties
            memid = MobTypeNode.create(self, type_name, (383, m))
            self.add_triple(subj=memid, pred_text="has_name", obj_text=type_name)
            if mob_name_to_properties.get(type_name) is not None:
                for property in mob_name_to_properties[type_name]:
                    self.add_triple(subj=memid, pred_text="has_name", obj_text=property)

    ##############
    ###  Mobs  ###
    ##############

    def set_mob_position(self, mob) -> "MobNode":
        r = self._db_read_one("SELECT uuid FROM ReferenceObjects WHERE eid=?", mob.entityId)
        if r:
            self._db_write(
                "UPDATE ReferenceObjects SET x=?, y=?, z=?, yaw=?, pitch=? WHERE eid=?",
                mob.pos.x,
                mob.pos.y,
                mob.pos.z,
                mob.look.yaw,
                mob.look.pitch,
                mob.entityId,
            )
            (memid,) = r
        else:
            memid = MobNode.create(self, mob)
        return self.get_mem_by_id(memid)

    ####################
    ###  ItemStacks  ###
    ####################

    def set_item_stack_position(self, item_stack) -> "ItemStackNode":
        r = self._db_read_one("SELECT uuid FROM ReferenceObjects WHERE eid=?", item_stack.entityId)
        if r:
            self._db_write(
                "UPDATE ReferenceObjects SET x=?, y=?, z=? WHERE eid=?",
                item_stack.pos.x,
                item_stack.pos.y,
                item_stack.pos.z,
                item_stack.entityId,
            )
            (memid,) = r
        else:
            memid = ItemStackNode.create(self, item_stack)
        return self.get_mem_by_id(memid)

    ###############
    ###  Dances  ##
    ###############

    def add_dance(self, dance_fn, name=None, tags=[]):
        # a dance is movement determined as a sequence of steps, rather than by its destination
        return DanceNode.create(self, dance_fn, name=name, tags=tags)
