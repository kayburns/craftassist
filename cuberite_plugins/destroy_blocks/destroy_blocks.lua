-- Copyright (c) Facebook, Inc. and its affiliates.


PLUGIN = nil

function Initialize(Plugin)
    Plugin:SetName("RemoveBlocks")
    Plugin:SetVersion(1)

    PLUGIN = Plugin
    cPluginManager.BindCommand("/destroy", "", Destroy, " ~ Destroy specified blocks");

    LOG("Loaded plugin: destroy_blocks")
    return true
end

function Destroy(Split, Player)
    if ((#Split - 1) % 3 ~= 0) then
        -- There was not a multiple of 3 provided
        -- Sending the proper usage to player
        Player:SendMessage("Usage: /destroy x1 y1 z1 ... xN yN zN")
        return true
    end
    
    local CurrentWorld = Player:GetWorld()

    -- Helper function to destropy blocks at given block locations
    local function Destroy(World)
	for i = 0,((#Split - 2) / 3) do
            p = (i * 3) + 2
	    q = p + 1
	    r = q + 1
	    World:DigBlock(Split[p], Split[q], Split[r])
	end
        -- for p = Split[2],Split[5] do
        --     for q = Split[3],Split[6] do
        --         for r = Split[4],Split[7] do
        --             World:DigBlock(p, q, r)
        --         end
        --     end
        -- end
    end

    CurrentWorld:ScheduleTask(0, Destroy)
 
    return true  
end
