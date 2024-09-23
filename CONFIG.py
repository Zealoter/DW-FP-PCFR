

test_sampling_train_config = {
    'CFR'     : {
        'game'      : None,
        'rm_mode'   : 'vanilla',
        'is_rm_plus': False,
        'ave_mode'  : 'liner',
    },
    'CFR+'    : {
        'game'      : None,
        'rm_mode'   : 'vanilla',
        'is_rm_plus': True,
        'ave_mode'  : 'liner',
    },
    'PCFR'    : {
        'game'      : None,
        'rm_mode'   : 'br',
        'is_rm_plus': False,
        'ave_mode'  : 'vanilla',
        'op_env'    : 'PCFR'
    },
    'DW-PCFR': {
        'game'      : None,
        'rm_mode'   : 'br',
        'is_rm_plus': False,
        'ave_mode'  : 'vanilla',
        'op_env'    : 'syncPCFR'
    },

}
