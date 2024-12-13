"""
    Test module for (very) simple test
    Essentially test passes if loss is not nan and not exceptions are thrown

    Reinforcement learning optimization
"""

import json
import numpy as np

from lib.systems import PulseShapingAWGNwithBWL, JointTxRxAWGNwithBWL,\
                        PulseShapingIM, JointTxRxIM, \
                        PulseShapingIMwithWDM, JointTxRxIMwithWDM

NUMPY_SEED = 1235246
N_SYMS = int(1e4)
CONSTELLATION = np.array([-3.0, -1.0, 1.0, 3.0])

def generate_symbols(n_syms, constellation):
    np_random_obj = np.random.default_rng(NUMPY_SEED)
    return np_random_obj.choice(constellation, (n_syms,), replace=True)

generate_symbols.__test__ = False  # do not test above function (only used for generating symbols)


def read_json_config(path_to_json: str):
    with open(path_to_json, 'r') as fp:
            config_dict = json.load(fp)
    return config_dict

read_json_config.__test__ = False  # do not test above function


def test_black_smoke_awgn_with_bwl_ps_reinforce():
    """
        Test PulseShaping for AWGN channel with bandwidth limitation
    """
    AWGN_CONFIG = 'test/test_config/awgn_bwl_config.json'
    RL_CONFIG = 'test/test_config/reinforce_config.json'
    awgn_kwargs = read_json_config(AWGN_CONFIG)
    rl_kwargs = read_json_config(RL_CONFIG)

    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = PulseShapingAWGNwithBWL(constellation=CONSTELLATION,
                                      tx_optimizer_params=rl_kwargs,
                                      **awgn_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))

def test_black_smoke_awgn_with_bwl_joint_reinforce():
    """
        Test JointRxTx for AWGN channel with bandwidth limitation
    """
    AWGN_CONFIG = 'test/test_config/awgn_bwl_config.json'
    RL_CONFIG = 'test/test_config/reinforce_config.json'
    awgn_kwargs = read_json_config(AWGN_CONFIG)
    rl_kwargs = read_json_config(RL_CONFIG)

    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = JointTxRxAWGNwithBWL(constellation=CONSTELLATION,
                                      tx_optimizer_params=rl_kwargs,
                                      **awgn_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_imdd_ps_reinforce():
    """
        Test PS for IM/DD channel
    """
    IMDD_CONFIG = 'test/test_config/imdd_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)
    RL_CONFIG = 'test/test_config/reinforce_config.json'
    rl_kwargs = read_json_config(RL_CONFIG)


    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = PulseShapingIM(constellation=CONSTELLATION,
                                tx_optimizer_params=rl_kwargs,
                                **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_imdd_joint_reinforce():
    """
        Test joint PS+RxF for IM/DD channel
    """
    IMDD_CONFIG = 'test/test_config/imdd_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)
    RL_CONFIG = 'test/test_config/reinforce_config.json'
    rl_kwargs = read_json_config(RL_CONFIG)


    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = JointTxRxIM(constellation=CONSTELLATION,
                             tx_optimizer_params=rl_kwargs,
                             **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_imdd_wdm_joint_reinforce():
    """
        Test joint PS+RxF for IM/DD channel with WDM
    """
    IMDD_CONFIG = 'test/test_config/imdd_wdm_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)
    RL_CONFIG = 'test/test_config/reinforce_config.json'
    rl_kwargs = read_json_config(RL_CONFIG)


    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = JointTxRxIMwithWDM(constellation=CONSTELLATION,
                                    tx_optimizer_params=rl_kwargs,
                                    **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_imdd_joint_reinforce_ssfm():
    """
        Test joint PS+RxF for IM/DD channel with RL and SSFM
    """
    IMDD_CONFIG = 'test/test_config/imdd_ssfm_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)
    RL_CONFIG = 'test/test_config/reinforce_config.json'
    rl_kwargs = read_json_config(RL_CONFIG)


    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = JointTxRxIM(constellation=CONSTELLATION,
                             tx_optimizer_params=rl_kwargs,
                             **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


