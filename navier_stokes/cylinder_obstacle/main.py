from navier_stokes.cylinder_obstacle.solver import CylinderPINN
from navier_stokes.cylinder_obstacle.utils import plot_flow_fields_cylinder

def main():
    R = 0.05
    nu = 0.01
    u_max = 1.0
    h_max = 0.2
    
    print("=== Cylinder PINN Simulator ===")
    pinn_solver = CylinderPINN(R=R, nu=nu, u_max=u_max, h_max=h_max)
    
    pinn_solver.build_and_train(
        iter_adam1=10000, 
        iter_adam2=15000, 
        save_path="pinn_cylinder_model"
    )

    print("\n=== Generating Flow Fields Plot ===")
    samples = pinn_solver.geom.random_points(500000)
    result = pinn_solver.predict(samples)

    plot_flow_fields_cylinder(samples, result, R=R)

if __name__ == "__main__":
    main()