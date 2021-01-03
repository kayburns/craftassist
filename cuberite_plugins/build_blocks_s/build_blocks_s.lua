-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

function Initialize(Plugin)
    Plugin:SetName("BuildBlocks")
    Plugin:SetVersion(1)

    PLUGIN = Plugin
    cPluginManager.BindCommand("/build", "", Build, " ~ Build specified blocks");

    LOG("Loaded plugin: build_blocks")
    return true
end

function Build(Split, Player)
    if ((#Split - 2) % 3 ~= 0) then
        -- There was not a multiple of 3 provided
        -- Sending the proper usage to player
        Player:SendMessage("Usage: /build b1 x1 y1 z1 ... bN xN yN zN")
        return true
    end
    
    local CurrentWorld = Player:GetWorld()

    -- Helper function to destropy blocks at given block locations
    local function Build(World)
	for i = 0,((#Split - 2) / 4) do
        b = (i * 4) + 2
	    p = b + 1
	    q = p + 1
	    r = q + 1
	    World:SetBlock(Split[p], Split[q], Split[r], Split[b], 0)
	end
    end

    CurrentWorld:ScheduleTask(0, Build)
 
    return true  
end
