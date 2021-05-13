import os
import gzip
import glob
import argparse

from collections import defaultdict

# to collect:
# - num blocks placed in sess1 and sess2 [3a] (DOUBLE CHECK THIS)
# + num commands from user in sess1 and sess2 [3b] 
# + parse (unparse, core, induced) [1]
# - total blocks placed, user block changes [2]
# - IP

def parse_agent_log(log_dir, agent_log_file='agent.log'):
    results = defaultdict(int)
    results['parse_type'] = []
    parse_was_decomposed = False
    with open(os.path.join(log_dir, agent_log_file)) as f:
        for line in f:
            if "Decomposing" in line:
                parse_was_decomposed = True
            if "PLAYER PLACED" in line:
                results['num_user_blocks'] += 1
            if "INFO]: Sending chat: /destroy" in line:
                block_count = len(line.split('destroy')[-1].split()) / 3
                results['num_agent_blocks'] += block_count
            if "INFO]: Sending chat: /build" in line:
                results['num_agent_blocks'] += 1
            if "INFO]: logical form post-coref" in line:
                if "DESTROY" in line or "BUILD" in line:
                    results['num_user_commands'] += 1
                    if parse_was_decomposed:
                        results['parse_type'].append("induced")
                        parse_was_decomposed = False
                    else:
                        results['parse_type'].append("core")
            if "I don't understand what you want me to destroy." in line:
                results['parse_type'].append("unparsable")
            if "I don't understand what you want me to build" in line:
                results['parse_type'].append("unparsable")
    return results

def decode_agent_mem(log_dir):
    #fname = glob.glob(os.path.join(log_dir, "agent_memory.bot.*.log.gz"))[0]
    fname = '/craftassist/agent_memory.bot.0102032.99.log.gz'
    with open(fname, 'rb') as f:
        mems = f.read()

    log_data = gzip.decompress(mems)
    log_data =  log_data.decode("utf-8").split(';')
    return log_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir', type=str, required=True,
        help='directory with logs to parse, unzipped from s3 download'
    )
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    #sess1_results = parse_agent_log(args.log_dir)
    #sess2_results = parse_agent_log(args.log_dir, agent_log_file='agent2.log')

    sess1_agentmem = decode_agent_mem(args.log_dir)

