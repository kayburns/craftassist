-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

function Initialize(Plugin)
    Plugin:SetName("PointSBlocks")
    Plugin:SetVersion(1)

    PLUGIN = Plugin
    cPluginManager.BindCommand("/point_s", "", PointS, " ~ Point at individually specified blocks");

    LOG("Loaded plugin: point_blocks_s")
    return true
end

function PointS(Split, Player)
    if ((#Split - 1) % 3 ~= 0) then
        -- Incorrect argument number
        -- Sending the proper usage to player
        Player:SendMessage("Usage: /point_s x1 y1 z1 ... xN yN zN")
        return true
    end
    
    local CurrentWorld = Player:GetWorld()

    -- 1. Get existing block-numbers and block-metas
    block_id_arr = {}
    block_meta_arr = {}
    count = 0

    for i = 0,((#Split - 2) / 3) do
	p = (i * 3) + 2
	q = p + 1
	r = q + 1
	block_id_arr[count] = CurrentWorld:GetBlock(Split[p], Split[q], Split[r])
	block_meta_arr[count] = CurrentWorld:GetBlockMeta(Split[p], Split[q], Split[r])
	count = count + 1
    end
   
    -- Helper function to set blocks at given block locations
    local function Build(World)
        for i = 0,((#Split - 2) / 3) do
    	    p = (i * 3) + 2
	    q = p + 1
	    r = q + 1
	    -- yellow stained glass has id 95:4
	    World:SetBlock(Split[p], Split[q], Split[r], 95, 4)
        end
    end


    -- Helper function to destropy blocks at given block locations
    local function Destroy(World)
        for i = 0,((#Split - 2) / 3) do
    	    p = (i * 3) + 2
	    q = p + 1
	    r = q + 1
	    World:DigBlock(Split[p], Split[q], Split[r])
        end
    end

    -- Helper function to restore the original state of the blocks
    local function Restore(World)
        count = 0
        for i = 0,((#Split - 2) / 3) do
    	    p = (i * 3) + 2
	    q = p + 1
	    r = q + 1
	    World:SetBlock(Split[p], Split[q], Split[r], block_id_arr[count], block_meta_arr[count])
	    count = count + 1
        end
    end

    -- 2. Perform build-and-destroy to mimic flickering
    num_flickers = 10 
    a = -1
    b = -1
    for i = 0,num_flickers do 
        a = i * 7
        b = a + 4

        CurrentWorld:ScheduleTask(a, Build)
        CurrentWorld:ScheduleTask(b, Destroy)
    end
 
    -- 3. Restore original state
    CurrentWorld:ScheduleTask(b + 1, Restore)
 
    return true  
end
