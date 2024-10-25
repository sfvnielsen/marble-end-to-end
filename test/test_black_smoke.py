"""
    Test module for (very) simple test
    Essentially test passes if loss is not nan and not exceptions are thrown
"""

import json
import numpy as np

from lib.systems import PulseShapingAWGN, RxFilteringAWGN, PulseShapingAWGNwithBWL,\
                        JointTxRxAWGNwithBWL, PulseShapingIM, RxFilteringIM, LinearFFEIM,\
                        JointTxRxIM,\
                        JointTxRxAWGNwithBWLandWDM, JointTxRxIMwithWDM

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


def test_black_smoke_awgn_ps():
    """
        Test pulse-shaper for AWGN channel
    """
    AWGN_CONFIG = 'test/test_config/awgn_config.json'
    awgn_kwargs = read_json_config(AWGN_CONFIG)

    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = PulseShapingAWGN(constellation=CONSTELLATION,
                                   **awgn_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_awgn_rxf():
    """
        Test RxFiltering for AWGN channel
    """
    AWGN_CONFIG = 'test/test_config/awgn_config.json'
    awgn_kwargs = read_json_config(AWGN_CONFIG)

    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = RxFilteringAWGN(constellation=CONSTELLATION,
                                   **awgn_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_awgn_with_bwl_ps():
    """
        Test PulseShaping for AWGN channel with bandwidth limitation
    """
    AWGN_CONFIG = 'test/test_config/awgn_bwl_config.json'
    awgn_kwargs = read_json_config(AWGN_CONFIG)

    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = PulseShapingAWGNwithBWL(constellation=CONSTELLATION,
                                          **awgn_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_awgn_with_bwl_joint():
    """
        Test JointRxTx for AWGN channel with bandwidth limitation
    """
    AWGN_CONFIG = 'test/test_config/awgn_bwl_config.json'
    awgn_kwargs = read_json_config(AWGN_CONFIG)

    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = JointTxRxAWGNwithBWL(constellation=CONSTELLATION,
                                       **awgn_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_imdd_ps():
    """
        Test PulseShaper for IM/DD channel
    """
    IMDD_CONFIG = 'test/test_config/imdd_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)

    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = PulseShapingIM(constellation=CONSTELLATION,
                                **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_imdd_rxf():
    """
        Test RxFiltering for IM/DD channel
    """
    IMDD_CONFIG = 'test/test_config/imdd_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)

    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = RxFilteringIM(constellation=CONSTELLATION,
                               **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_imdd_joint():
    """
        Test JointTxRx for IM/DD channel
    """
    IMDD_CONFIG = 'test/test_config/imdd_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)

    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = JointTxRxIM(constellation=CONSTELLATION,
                             **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_imdd_ffe():
    """
        Test FFE for IM/DD channel
    """
    IMDD_CONFIG = 'test/test_config/imdd_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)
    FFE_TAPS = 25

    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = LinearFFEIM(constellation=CONSTELLATION,
                             ffe_n_taps=FFE_TAPS,
                             **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_awgn_wdm_joint():
    """
        Test JointTxRx for AWGN channel with BWL and WDM
    """
    IMDD_CONFIG = 'test/test_config/awgn_bwl_wdm_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)

    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = JointTxRxAWGNwithBWLandWDM(constellation=CONSTELLATION,
                                            **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))


def test_black_smoke_imdd_wdm_joint():
    """
        Test JointTxRx for IM/DD channel with WDM
    """
    IMDD_CONFIG = 'test/test_config/imdd_wdm_config.json'
    imdd_kwargs = read_json_config(IMDD_CONFIG)

    syms = generate_symbols(N_SYMS, CONSTELLATION)

    e2e_system = JointTxRxIMwithWDM(constellation=CONSTELLATION,
                                    **imdd_kwargs)
    
    e2e_system.initialize_optimizer()

    loss = e2e_system.optimize(syms, return_loss=True)
    __ = e2e_system.evaluate(syms)

    assert np.all(np.logical_not(np.isnan(loss)))