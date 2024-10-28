"""
    Test module for (very) simple test
    Essentially test passes if loss is not nan and not exceptions are thrown

    Only for surrogate models
"""

import json
import numpy as np

from lib.systems import JointTxRxAWGNwithBWL, PulseShapingIM, JointTxRxIM

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


def test_black_smoke_awgn_with_bwl_joint_surrogate():
    """
        Test JointRxTx for AWGN channel with bandwidth limitation
    """
    AWGN_CONFIG = 'test/test_config/awgn_bwl_config.json'
    SURROGATE_CONFIG = 'test/test_config/surrogate_config.json'
    awgn_kwargs = read_json_config(AWGN_CONFIG)
    surrogate_kwargs = read_json_config(SURROGATE_CONFIG)

    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = JointTxRxAWGNwithBWL(constellation=CONSTELLATION,
                                       tx_optimizer_params=surrogate_kwargs,
                                       **awgn_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))

def test_black_smoke_imdd_ps_surrogate():
    """
        Test PS for IM/DD channel with surrogate
    """
    IMDD_CONFIG = 'test/test_config/imdd_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)
    SURROGATE_CONFIG = 'test/test_config/surrogate_config.json'
    surrogate_kwargs = read_json_config(SURROGATE_CONFIG)


    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = PulseShapingIM(constellation=CONSTELLATION,
                                tx_optimizer_params=surrogate_kwargs,
                                **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_imdd_joint_surrogate():
    """
        Test joint PS+RxF for IM/DD channel with surrogate
    """
    IMDD_CONFIG = 'test/test_config/imdd_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)
    SURROGATE_CONFIG = 'test/test_config/surrogate_config.json'
    surrogate_kwargs = read_json_config(SURROGATE_CONFIG)


    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = JointTxRxIM(constellation=CONSTELLATION,
                             tx_optimizer_params=surrogate_kwargs,
                             **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_imdd_ps_surrogate_indep_tx_optim():
    """
        Test PS for IM/DD channel with surrogate (independent optimizers)
    """
    IMDD_CONFIG = 'test/test_config/imdd_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)
    SURROGATE_CONFIG = 'test/test_config/surrogate_config_independent_optimizers.json'
    surrogate_kwargs = read_json_config(SURROGATE_CONFIG)


    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = PulseShapingIM(constellation=CONSTELLATION,
                                tx_optimizer_params=surrogate_kwargs,
                                **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_imdd_joint_surrogate_only():
    """
        Test joint PS+RxF for IM/DD channel with surrogate
    """
    IMDD_CONFIG = 'test/test_config/imdd_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)
    SURROGATE_CONFIG = 'test/test_config/surrogate_config.json'
    surrogate_kwargs = read_json_config(SURROGATE_CONFIG)


    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = JointTxRxIM(constellation=CONSTELLATION,
                             tx_optimizer_params=surrogate_kwargs,
                             **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize_surrogate(syms, return_loss=True)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_imdd_joint_surrogate_whnn_only():
    """
        Test joint PS+RxF for IM/DD channel with surrogate
    """
    IMDD_CONFIG = 'test/test_config/imdd_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)
    SURROGATE_CONFIG = 'test/test_config/surrogate_config_wh.json'
    surrogate_kwargs = read_json_config(SURROGATE_CONFIG)


    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = JointTxRxIM(constellation=CONSTELLATION,
                             tx_optimizer_params=surrogate_kwargs,
                             **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize_surrogate(syms, return_loss=True)

    assert np.all(np.logical_not(np.isnan(loss)))

def test_black_smoke_imdd_joint_surrogate_eval_grad_error():
    """
        Test joint PS+RxF for IM/DD channel with surrogate
        Test the evaluate_gradient_error method
    """
    IMDD_CONFIG = 'test/test_config/imdd_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)
    SURROGATE_CONFIG = 'test/test_config/surrogate_config_wh.json'
    surrogate_kwargs = read_json_config(SURROGATE_CONFIG)


    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = JointTxRxIM(constellation=CONSTELLATION,
                             tx_optimizer_params=surrogate_kwargs,
                             **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    e2e_system.optimize_surrogate(syms)

    grad_error = e2e_system.evaluate_tx_gradient_error(syms)

    assert not np.isnan(grad_error)