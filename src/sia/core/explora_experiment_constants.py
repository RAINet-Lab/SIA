ENV_KPI_NAME_LIST = ['tx_brate', 'tx_pckts', 'dl_buffer']

# The association of folder number with agents
AGENT_EXPERIMENT_INFO = {
    "embb-trf1": {
        "name": "embb-trf1",
        "experiment_directories": [1,2,3,4,5,6,7,8],
        "num_of_users": {
            1:6,
            2:5,
            3:4,
            4:3,
            5:2,
            6:1,
            7:1,
            8:1
        },
        "exp_for_user_num": {
            1: [6, 7, 8],
            2: [5],
            3: [4],
            4: [3],
            5: [2],
            6: [1]
        }
    },
    "embb-trf2":{
        "name": "embb-trf2",
        "experiment_directories": [9,10,11,12,13,14,15,16],
        "num_of_users": {
            9:6,
            10:5,
            11:4,
            12:3,
            13:2,
            14:1,
            15:1,
            16:1
        },
        "exp_for_user_num": {
            1: [14, 15, 16],
            2: [13],
            3: [12],
            4: [11],
            5: [10],
            6: [9]
        }
    },
    "urllc-trf1":{
        "name": "urllc-trf1",
        "experiment_directories": [27,28,29,30,31,32,33,34],
        "num_of_users": {
            27:6,
            28:5,
            29:4,
            30:3,
            31:2,
            32:1,
            33:1,
            34:1
        },
        "exp_for_user_num": {
            1: [32, 33, 34],
            2: [31],
            3: [30],
            4: [29],
            5: [28],
            6: [27]
        }
    },
    "urllc-trf2":{
        "name": "urllc-trf2",
        "experiment_directories": [35,36,37,38,39,40,41,42],
        "num_of_users": {
            35:6,
            36:5,
            37:4,
            38:3,
            39:2,
            40:1,
            41:1,
            42:1
        },
        "exp_for_user_num": {
            1: [40, 41, 42],
            2: [39],
            3: [38],
            4: [37],
            5: [36],
            6: [35]
        }
    }  
}