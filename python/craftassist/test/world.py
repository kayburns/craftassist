import numpy as np
from typing import Sequence, Dict
from util import Block, XYZ, IDM
from utils import Pos, Look
from rotation import look_vec
from entities import MOBS_BY_ID

FLAT_GROUND_DEPTH = 8
FALL_SPEED = 1
# TODO make these prettier
MOB_COLORS = {
    "rabbit": (0.3, 0.3, 0.3),
    "cow": (0.9, 0.9, 0.9),
    "pig": (0.9, 0.5, 0.5),
    "chicken": (0.9, 0.9, 0.0),
    "sheep": (0.6, 0.6, 0.6),
}
MOB_META = {101: "rabbit", 92: "cow", 90: "pig", 93: "chicken", 91: "sheep"}
MOB_SPEED = {"rabbit": 1, "cow": 0.3, "pig": 0.5, "chicken": 1, "sheep": 0.3}
MOB_LOITER_PROB = {"rabbit": 0.3, "cow": 0.5, "pig": 0.3, "chicken": 0.1, "sheep": 0.5}
MOB_LOITER_TIME = {"rabbit": 2, "cow": 2, "pig": 2, "chicken": 1, "sheep": 2}
MOB_STEP_HEIGHT = {"rabbit": 1, "cow": 1, "pig": 1, "chicken": 2, "sheep": 2}
MOB_DIRECTION_CHANGE_PROB = {"rabbit": 0.1, "cow": 0.1, "pig": 0.2, "chicken": 0.3, "sheep": 0.2}


class Opt:
    pass


class MobInfo:
    pass


def flat_ground_generator_with_grass(world):
    flat_ground_generator(world, grass=True)


def flat_ground_generator(world, grass=False):
    r = world.to_world_coords((0, 62, 0))[1]
    # r = world.sl // 2
    world.blocks[:] = 0
    world.blocks[:, 0:r, :, 0] = 7
    world.blocks[:, r - FLAT_GROUND_DEPTH : r, :, 0] = 3
    if grass:
        world.blocks[:, r, :, 0] = 2
    else:
        world.blocks[:, r, :, 0] = 3


def make_mob_opts(mobname):
    opts = Opt()
    opts.mobname = mobname
    opts.direction_change_prob = MOB_DIRECTION_CHANGE_PROB[mobname]
    opts.color = MOB_COLORS[mobname]
    opts.speed = MOB_SPEED[mobname]
    opts.loiter_prob = MOB_LOITER_PROB[mobname]
    opts.loiter_time = MOB_LOITER_TIME[mobname]
    opts.step_height = MOB_STEP_HEIGHT[mobname]
    opts.mobType = list(MOBS_BY_ID.keys())[list(MOBS_BY_ID.values()).index(mobname)]
    return opts


def check_bounds(p, sl):
    if p >= sl or p < 0:
        return -1
    return 1


class SimpleMob:
    def __init__(self, opts, start_pos=None, start_look=(0.0, 0.0)):
        self.mobname = opts.mobname
        self.color = opts.color
        self.direction_change_prob = opts.direction_change_prob
        self.loiter_prob = opts.loiter_prob
        self.loiter_time = opts.loiter_time
        self.speed = opts.speed
        self.step_height = opts.step_height
        self.pos = start_pos
        self.look = start_look
        self.loitering = -1
        self.new_direction()
        self.entityId = str(np.random.randint(0, 100000))
        self.mobType = opts.mobType
        self.agent_built = False

    def add_to_world(self, world):
        self.world = world
        if self.pos is None:
            xz = np.random.randint(0, world.sl, (2,))
            slice = self.world.blocks[xz[0], :, xz[1], 0]
            nz = np.flatnonzero(slice)
            if len(nz) == 0:
                # mob will be floating, but why no floor here?
                h = 0
            else:
                # if top block is nonzero mob will be trapped
                h = nz[-1]
            off = self.world.coord_shift
            self.pos = (float(xz[0]) + off[0], float(h + 1) + off[1], float(xz[1]) + off[2])
        self.world.mobs.append(self)

    def get_info(self):
        info = MobInfo()
        info.entityId = self.entityId
        info.pos = Pos(*self.pos)
        info.look = Look(*self.look)
        info.mobType = self.mobType
        info.color = self.color
        info.mobname = self.mobname
        return info

    def new_direction(self):
        new_direction = np.random.randn(2)
        self.direction = new_direction / np.linalg.norm(new_direction)
        # self.look ##NOT IMPLEMENTED

    def step(self):
        # check if falling:
        x, y, z = self.world.to_world_coords(self.pos)
        fy = int(np.floor(y))
        rx = int(np.round(x))
        rz = int(np.round(z))
        if y > 0:
            if self.world.blocks[rx, fy - 1, rz, 0] == 0:
                self.pos = (self.pos[0], self.pos[1] - FALL_SPEED, self.pos[2])
                return
        # TODO when look implemented: change looks when loitering
        if self.loitering >= 0:
            self.loitering += 1
            if self.loitering > self.loiter_time:
                self.loitering = -1
            return
        if np.random.rand() < self.loiter_prob:
            self.loitering = 0
            return
        if np.random.rand() < self.direction_change_prob:
            self.new_direction()
        step = self.direction * self.speed
        bx = check_bounds(int(np.round(x + step[0])), self.world.sl)
        bz = check_bounds(int(np.round(z + step[1])), self.world.sl)
        # if hitting boundary, reverse...
        self.direction[0] = bx * self.direction[0]
        self.direction[1] = bz * self.direction[1]
        step = self.direction * self.speed
        new_x = step[0] + x
        new_z = step[1] + z
        # is there a block in new location? if no go there, if yes go up
        for i in range(self.step_height):
            if fy + i >= self.world.sl:
                self.new_direction()
                return
            if self.world.blocks[int(np.round(new_x)), fy + i, int(np.round(new_z)), 0] == 0:
                self.pos = self.world.from_world_coords((new_x, y + i, new_z))
                return
        # couldn't get past a wall of blocks, try a different dir
        self.new_direction()
        return


def shift_coords(p, shift):
    if hasattr(p, "x"):
        q = Opt()
        q.x = p.x + shift[0]
        q.y = p.y + shift[1]
        q.z = p.z + shift[2]
        return q
    q = np.add(p, shift)
    if type(p) is tuple:
        q = tuple(q)
    if type(p) is list:
        q = list(q)
    return q


def build_coord_shifts(coord_shift):
    def to_world_coords(p):
        dx = -coord_shift[0]
        dy = -coord_shift[1]
        dz = -coord_shift[2]
        return shift_coords(p, (dx, dy, dz))

    def from_world_coords(p):
        return shift_coords(p, coord_shift)

    return to_world_coords, from_world_coords


class World:
    def __init__(self, opts, spec):
        self.opts = opts
        self.count = 0
        self.sl = opts.sl

        # to be subtracted from incoming coordinates and added to outgoing
        self.coord_shift = spec.get("coord_shift", (0, 0, 0))
        to_world_coords, from_world_coords = build_coord_shifts(self.coord_shift)
        self.to_world_coords = to_world_coords
        self.from_world_coords = from_world_coords

        self.blocks = np.zeros((opts.sl, opts.sl, opts.sl, 2), dtype="int32")
        if spec.get("ground_generator"):
            spec["ground_generator"](self)
        else:
            self.build_ground()
        self.mobs = []
        for m in spec["mobs"]:
            m.add_to_world(self)
        self.item_stacks = []
        for i in spec["item_stacks"]:
            i.add_to_world(self)
        self.players = []
        for p in spec["players"]:
            self.players.append(p)
        self.agent_data = spec["agent"]
        self.chat_log = []

        # FIXME
        self.mob_opt_maker = make_mob_opts
        self.mob_maker = SimpleMob

        # TODO: Add item stack maker?

    def step(self):
        for m in self.mobs:
            m.step()
        self.count += 1

    def build_ground(self):
        if hasattr(self.opts, "avg_ground_height"):
            avg_ground_height = self.opts.avg_ground_height
        else:
            avg_ground_height = 6.0
        if hasattr(self.opts, "hill_scale"):
            hill_scale = self.opts.hill_scale
        else:
            hill_scale = 5.0
        p = hill_scale * np.random.randn(6)
        g = np.mgrid[0 : self.sl, 0 : self.sl].astype("float32") / self.sl
        ground_height = (
            p[0] * np.sin(g[0])
            + p[1] * np.cos(g[0])
            + p[2] * np.cos(g[0]) * np.sin(g[0])
            + p[3] * np.sin(g[1])
            + p[4] * np.cos(g[1])
            + p[5] * np.cos(g[1]) * np.sin(g[1])
        )
        ground_height = ground_height - ground_height.mean() + avg_ground_height
        for i in range(self.sl):
            for j in range(self.sl):
                for k in range(int(ground_height[i, j])):
                    self.blocks[i, k, j] = (3, 0)

        # FIXME this is broken
        if hasattr(self.opts, "ground_block_probs"):
            ground_blocks = np.transpose(np.nonzero(self.blocks[:, :, :, 0] == 3))
            num_ground_blocks = len(ground_blocks)
            for idm, val in self.opts.ground_block_probs:
                if idm != (3, 0):
                    num = np.random.rand() * val * 2 * num_ground_blocks
                    for i in range(num):
                        j = np.random.randint(num_ground_blocks)
                        self.blocks[
                            ground_blocks[j][0], ground_blocks[j][1], ground_blocks[j][2], :
                        ] = idm

    def place_block(self, block: Block, force=False):
        loc, idm = block
        if idm[0] == 383:
            # its a mob...
            try:
                # FIXME handle unknown mobs/mobs not in list
                m = SimpleMob(make_mob_opts(MOB_META[idm[1]]), start_pos=loc)
                m.agent_built = True
                m.add_to_world(self)
                return True
            except:
                return False
        # mobs keep loc in real coords, block objects we convert to the numpy index
        loc = tuple(self.to_world_coords(loc))
        if idm[0] == 0:
            try:
                if tuple(self.blocks[loc]) != (7, 0) or force:
                    self.blocks[loc] = (0, 0)
                    return True
                else:
                    return False
            except:
                return False
        else:
            try:
                if tuple(self.blocks[loc]) != (7, 0) or force:
                    self.blocks[loc] = idm
                    return True
                else:
                    return False
            except:
                return False

    def dig(self, loc: XYZ):
        return self.place_block((loc, (0, 0)))

    def blocks_to_dict(self):
        d = {}
        nz = np.transpose(self.blocks[:, :, :, 0].nonzero())
        for loc in nz:
            l = tuple(loc.tolist())
            d[self.from_world_coords(l)] = tuple(self.blocks[l[0], l[1], l[2], :])
        return d

    def get_idm_at_locs(self, xyzs: Sequence[XYZ]) -> Dict[XYZ, IDM]:
        """Return the ground truth block state"""
        d = {}
        for (x, y, z) in xyzs:
            B = self.get_blocks(x, x, y, y, z, z)
            d[(x, y, z)] = tuple(B[0, 0, 0, :])
        return d

    def get_mobs(self):
        return [m.get_info() for m in self.mobs]

    def get_item_stacks(self):
        return [i.get_info() for i in self.item_stacks]

    def get_blocks(self, xa, xb, ya, yb, za, zb, transpose=True):
        xa, ya, za = self.to_world_coords((xa, ya, za))
        xb, yb, zb = self.to_world_coords((xb, yb, zb))
        M = np.array((xb, yb, zb))
        m = np.array((xa, ya, za))
        szs = M - m + 1
        B = np.zeros((szs[1], szs[2], szs[0], 2), dtype="uint8")
        B[:, :, :, 0] = 7
        xs, ys, zs = [0, 0, 0]
        xS, yS, zS = szs
        if xb < 0 or yb < 0 or zb < 0:
            return B
        if xa > self.sl - 1 or ya > self.sl - 1 or za > self.sl - 1:
            return B
        if xb > self.sl - 1:
            xS = self.sl - xa
            xb = self.sl - 1
        if yb > self.sl - 1:
            yS = self.sl - ya
            yb = self.sl - 1
        if zb > self.sl - 1:
            zS = self.sl - za
            zb = self.sl - 1
        if xa < 0:
            xs = -xa
            xa = 0
        if ya < 0:
            ys = -ya
            ya = 0
        if za < 0:
            zs = -za
            za = 0
        pre_B = self.blocks[xa : xb + 1, ya : yb + 1, za : zb + 1, :]
        # pre_B = self.blocks[ya : yb + 1, za : zb + 1, xa : xb + 1, :]
        B[ys:yS, zs:zS, xs:xS, :] = pre_B.transpose(1, 2, 0, 3)
        if transpose:
            return B
        else:
            return pre_B

    def get_line_of_sight(self, pos, yaw, pitch):
        # it is assumed lv is unit normalized
        pos = tuple(self.to_world_coords(pos))
        lv = look_vec(yaw, pitch)
        dt = 1.0
        for n in range(2 * self.sl):
            p = tuple(np.round(np.add(pos, n * dt * lv)).astype("int32"))
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        sp = tuple(np.add(p, (i, j, k)))
                        if all([x >= 0 for x in sp]) and all([x < self.sl for x in sp]):
                            if tuple(self.blocks[sp]) != (0, 0):
                                # TODO: deal with close blocks artifacts,
                                # etc
                                return tuple(self.from_world_coords(sp))
        return

    def add_incoming_chat(self, chat: str, speaker_name: str):
        """Add a chat to memory as if it was just spoken by SPEAKER"""
        self.chat_log.append("<" + speaker_name + ">" + " " + chat)
