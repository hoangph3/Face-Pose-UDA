dataset_info = dict(
    dataset_name='ikemen',
    paper_info='',
    keypoint_info={
        0:
        dict(
            name='scale_bot',
            id=0,
            color=[255, 128, 0],
            type='',
            swap=''),
        1:
        dict(
            name='scale_center',
            id=1,
            color=[0, 255, 0],
            type='',
            swap=''),
        2:
        dict(
            name='scale_right',
            id=2,
            color=[255, 128, 0],
            type='',
            swap='scale_left'),
        3:
        dict(
            name='scale_left', 
            id=3, color=[51, 153, 255], 
            type='', 
            swap='scale_right'),
    },
    skeleton_info={},
    joint_weights=[1.] * 4,
    sigmas=[])