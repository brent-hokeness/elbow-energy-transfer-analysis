"""
Definition of kinematic variables to extract for analysis.
Based on pelvis, torso, shoulder, and elbow measurements.
"""

# Kinematic variables to extract at each time point
KINEMATIC_VARIABLES = {
    'elbow': [
        'elbow_angle_x',  # elbow flexion/extension
        'elbow_angle_y',  # elbow abduction/adduction  
        'elbow_angle_z'   # elbow rotation
    ],
    
    'shoulder': [
        'shoulder_angle_x',  # shoulder flexion/extension
        'shoulder_angle_y',  # shoulder abduction/adduction
        'shoulder_angle_z'   # shoulder rotation
    ],
    
    'torso': [
        'torso_angle_x',     # torso anterior/posterior tilt
        'torso_angle_y',     # torso lateral tilt  
        'torso_angle_z'      # torso rotation
    ],
    
    'pelvis': [
        'pelvis_angle_x',    # pelvis anterior/posterior tilt
        'pelvis_angle_y',    # pelvis lateral tilt
        'pelvis_angle_z'     # pelvis rotation
    ],
    
    'torso_pelvis': [
        'torso_pelvis_angle_x',  # torso-pelvis relative anterior/posterior
        'torso_pelvis_angle_y',  # torso-pelvis relative lateral
        'torso_pelvis_angle_z'   # torso-pelvis relative rotation
    ]
}

# All variables in a flat list
ALL_KINEMATIC_VARIABLES = []
for joint_group in KINEMATIC_VARIABLES.values():
    ALL_KINEMATIC_VARIABLES.extend(joint_group)

# Time points of interest
TIME_POINTS = {
    'foot_plant': 'FP_v6_time',
    'max_external_rotation': 'MER_time', 
    'max_internal_rotation': 'MIR_time',
    'ball_release': 'BR_time'
}