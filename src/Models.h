#include <Eigen/Dense>
#include "QuaternionTools.h"

#ifndef MODELS_H
#define MODELS_H

namespace KalmanFilter {
    // IMU-based process function
    static Eigen::VectorXd f_fun_imu(const Eigen::VectorXd& x, const Eigen::VectorXd& u_m, const double &dt){
  
        // x = [x_n y_e z_d v_x v_y v_z q_w q_x q_y q_z b_ax b_ay b_az b_wx b_wy b_wz]
        Eigen::VectorXd f(x.rows());
        f.setZero();
        Eigen::VectorXd g_(3);
        g_<< 0,0,9.81;
        
        // Measurement minus bias and noise
        Eigen::VectorXd u(u_m.rows());
        Eigen::VectorXd noise(6,1);
        noise.setZero();
        noise.block(3,0,3,1) = x.block(16,0,3,1);
        u = u_m - x.block(10,0,6,1) - noise;
        
        double e_w, e_x, e_y, e_z; 
        e_w = x[0]; e_x = x[1]; e_y = x[2]; e_z = x[3];

        double p, q, r; // girometer measurements
        p = u[3]; q = u[4]; r = u[5];

        // Create a rotation matrix from body- to inertial-frame

        //   Eigen::MatrixXd R_bn(3,3); // Rotation matrix body-frame to navigation-frame
        //   R_bn<< pow(e_x,2)+pow(e_w,2)-pow(e_y,2)-pow(e_z,2), -2*e_z*e_w+2*e_y*e_x, 2*e_y*e_w +2*e_z*e_x,
        // 	        2*e_x*e_y+ 2*e_w*e_z, pow(e_y,2)+pow(e_w,2)-pow(e_x,2)-pow(e_z,2), 2*e_z*e_y-2*e_x*e_w,
        // 	        2*e_x*e_z-2*e_w*e_y, 2*e_y*e_z+2*e_w*e_x, pow(e_z,2)+pow(e_w,2)-pow(e_x,2)-pow(e_y,2);
        
        Eigen::Quaterniond q_b;
        q_b.x() = e_x;
        q_b.y() = e_y;
        q_b.z() = e_z;
        q_b.w() = e_w; 

        Eigen::Matrix3d R_bn(3,3);
        R_bn = q_b.normalized().toRotationMatrix();
            
        // Create a skew-symetric matrix
        Eigen::MatrixXd omega(3,3);
        omega << 0, -r, q,
                    r, 0, -p,
                -q, p, 0;
            
        // Orientation propagation
        Eigen::MatrixXd Omega(4,4);
        Omega << 0, -p, -q, -r,
                p, 0, r, -q,
                q, -r, 0, p,
                r, q, -p, 0;

        double s, sin_s, norm_u;
        norm_u = sqrt(pow(p,2) + pow(q,2) + pow(r,2));
        s = 0.5*dt*norm_u;
        if (s > 0){
            sin_s = sin(s)/s;
        }
        else {
            sin_s = 1;
        }
        Eigen::MatrixXd M(4,4);
        M.setIdentity();
        M = cos(s)*M + (0.5*dt*sin_s)*Omega;
        
        // Position 
        f.block(4,0,3,1) = x.block(4,0,3,1) + dt*R_bn*x.block(7,0,3,1); 

            // Velocity in the body reference-frame
        f.block(7,0,3,1) = x.block(7,0,3,1) + dt*(-omega*x.block(7,0,3,1) + u.block(0,0,3,1) + R_bn.transpose()*g_); 

        //Orientation propagation
        f.block(0,0,4,1) = M*x.block(0,0,4,1);
        
        //Bias propagation
        f.block(10,0,6,1) = x.block(10,0,6,1);
        
        return f;
    }

    //Observation model based on position [x_north y_east]
    static Eigen::VectorXd h_fun_pos(const Eigen::VectorXd& x){  
        return x.block(4,0,2,1);
    }

    //Observation model based on altitude [H]
    Eigen::VectorXd h_fun_bar(const Eigen::VectorXd& x){
        return -x.block(6,0,1,1);
    }

    //Observation model based on AHRS [q]
    Eigen::VectorXd h_fun_ahrs(const Eigen::VectorXd& x){  
        return x.block(0,0,4,1);
    }

    //Observation model based on Pitot [V]
    Eigen::VectorXd h_fun_pitot(const Eigen::VectorXd& x){ 
        Eigen::VectorXd h(1,1);
        h.setZero();
        h[0] = x.block(7,0,3,1).norm();
        return h;
    }

    //Observation model based GPS velocity
    Eigen::VectorXd h_fun_gps_vel(const Eigen::VectorXd& x){ 
        Eigen::VectorXd h(2,1);
        h.setZero();

        // Create a rotation matrix
        Eigen::Quaterniond q_b;
        q_b.x() = x[0];
        q_b.y() = x[1];
        q_b.z() = x[2];
        q_b.w() = x[3]; 

        Eigen::Matrix3d R_bn(3,3);
        R_bn = q_b.normalized().toRotationMatrix();
        h = R_bn.block(0,0,2,2)*x.block(7,0,2,1);

        return h;
    }
}
#endif // MODELS_H