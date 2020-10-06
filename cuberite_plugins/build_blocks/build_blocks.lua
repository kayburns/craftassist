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
        Player:SendMessage("Usage: /build b x1 y1 z1 ... xN yN zN")
        return true
    end
    
    local CurrentWorld = Player:GetWorld()

    -- Helper function to destropy blocks at given block locations
    local function Build(World)
	b = Split[2]
	for i = 0,((#Split - 3) / 3) do
            p = (i * 3) + 3
	    q = p + 1
	    r = q + 1
	    World:SetBlock(Split[p], Split[q], Split[r], b, 0)
	end
    end

    CurrentWorld:ScheduleTask(0, Build)
 
    return true  
end
