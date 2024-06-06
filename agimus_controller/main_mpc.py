import numpy as np

from agimus_controller.ocp import OCPPandaReachingColWithMultipleCol
from agimus_controller.wrapper_panda import PandaWrapper
from agimus_controller.scenes import Scene
from agimus_controller.mpc import MPC


def main():
    # # # # # # # # # # # # # # # # # # #
    ### LOAD ROBOT MODEL and SIMU ENV ###
    # # # # # # # # # # # # # # # # # # #

    # Name of the scene (can be changed by "ball" and "wall")
    name_scene = "box"

    # Pose of the obstacle

    # Creation of the scene
    scene = Scene(name_scene=name_scene)
    
    robotwrapper = PandaWrapper(capsule=True, auto_col=True)
    rmodel, cmodel, vmodel = robotwrapper.create_robot()
    rmodel, cmodel, TARGET_POSE1, TARGET_POSE2, q0 = scene.create_scene_from_urdf(rmodel, cmodel)

    # Extract robot model
    nv = rmodel.nv
    v0 = np.zeros(nv)
    x0 = np.concatenate([q0, v0])
    
    # Parameters of the OCP
    max_iter = 4  # Maximum iterations of the solver
    max_qp_iters = 25  # Maximum iterations for solving each qp solved in one iteration of the solver
    dt = 2e-2
    T = 10
    WEIGHT_GRIPPER_POSE = 1e2
    WEIGHT_xREG = 1e-2
    WEIGHT_uREG = 1e-4
    max_qp_iters = 25
    callbacks = False
    safety_threshhold = 7e-2

    # Parameters of the MPC
    T_sim = 0.5

    ### CREATING THE PROBLEM WITH OBSTACLE

    print("Solving the problem with collision")
    problem = OCPPandaReachingColWithMultipleCol(
        rmodel,
        cmodel,
        TARGET_POSE1,
        T,
        dt,
        x0,
        WEIGHT_xREG=WEIGHT_xREG,
        WEIGHT_uREG=WEIGHT_uREG,
        WEIGHT_GRIPPER_POSE=WEIGHT_GRIPPER_POSE,
        MAX_QP_ITERS=max_qp_iters,
        SAFETY_THRESHOLD=safety_threshhold,
        callbacks=callbacks,
    )

    mpc = MPC(
        robot_simulator,
        OCP=problem,
        max_iter=max_iter,
        TARGET_POSE_1=TARGET_POSE1,
        TARGET_POSE_2=TARGET_POSE2,
        T_sim=T_sim,
    )
    mpc.solve()
    mpc.plot_collision_distances()
    mpc.plot_mpc_results()


if __name__ == "__main__":
    main()
