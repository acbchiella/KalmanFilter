#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#ifndef QUATERNIONTOOLS_h
#define QUATERNIONTOOLS_h

namespace KalmanFilter {
    // Quaternion multiplication
    static Eigen::VectorXd quat_multiply(const Eigen::VectorXd &q1, const Eigen::VectorXd &q2){
        Eigen::VectorXd q(4,1);
        q(0,0) = q2(0,0)*q1(0,0) - q2(1,0)*q1(1,0) - q2(2,0)*q1(2,0) - q2(3,0)*q1(3,0);
        q(1,0) = q2(0,0)*q1(1,0) + q2(1,0)*q1(0,0) - q2(2,0)*q1(3,0) + q2(3,0)*q1(2,0);
        q(2,0) = q2(0,0)*q1(2,0) + q2(1,0)*q1(3,0) + q2(2,0)*q1(0,0) - q2(3,0)*q1(1,0);
        q(3,0) = q2(0,0)*q1(3,0) - q2(1,0)*q1(2,0) + q2(2,0)*q1(1,0) + q2(3,0)*q1(0,0);
        
        return q;
    }
    // Unit quaternion to rotation transform
    static Eigen::VectorXd q2r(Eigen::VectorXd &q){
        Eigen::VectorXd r(3,1);
        r.setZero();
        
        q = q/q.norm();
        double theta = 0;

        if (q(0,0)>=0){
            theta = 2*acos(q(0,0));   
        } else {
            theta = -2*acos(-q(0,0));
        }

        double norma = q.block(1,0,3,1).norm();
        if(norma>0.000000000000000000001){
            r = theta*q.block(1,0,3,1)/norma;
        }else{
            r = q.block(1,0,3,1);
        }

        return r;
    }
    // Rotation vector to unit quaternion transform.
    static Eigen::VectorXd r2q(const Eigen::VectorXd &r){
        Eigen::VectorXd q(4,1);
        q.setZero();

        double theta = r.norm();
        if (theta>0){
            q(0,0) = cos(0.5*theta);
            q.block(1,0,3,1) = sin(0.5*theta)*r/theta;
        }else{
            q(0,0) = 1;
            q.block(1,0,3,1).setZero();
        }

        return q;
    }
    // Modified minus operation
    Eigen::VectorXd o_minus(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2){
    int nx_1 = x1.rows();
    
        if (nx_1>4){
            Eigen::VectorXd x(nx_1-1,1);
            x.setZero();
            Eigen::VectorXd q_inv(4,1);
            // The inverse of unit quaternion is equals to its conjugate
            q_inv(0,0) = x2(0,0);
            q_inv.block(1,0,3,1) = - x2.block(1,0,3,1);
            
            Eigen::VectorXd q_multiply(4,1);
            q_multiply = quat_multiply(x1.block(0,0,4,1), q_inv);

            x.block(0,0,3,1) = q2r(q_multiply);
            x.block(3,0,nx_1-4,1) = x1.block(4,0,nx_1-4,1) - x2.block(4,0,nx_1-4,1);
            return x;
        }
        if (nx_1==4){
            Eigen::VectorXd x(nx_1-1,1);
            x.setZero();
            Eigen::VectorXd q_inv(4,1);
            // The inverse of unit quaternion is equals to its conjugate
            q_inv(0,0) = x2(0,0);
            q_inv.block(1,0,3,1) = - x2.block(1,0,3,1);
            
            Eigen::VectorXd q_multiply(4,1);
            q_multiply = quat_multiply(x1, q_inv);
            x = q2r(q_multiply);
            return x;
        }
        if (nx_1<4){
            Eigen::VectorXd x(nx_1,1);
            x.setZero();
        
            x = x1 - x2;  
            return x;
        }
    }
    // Modified sum operation.
    static Eigen::VectorXd o_sum(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2){

        int nx_1 = x1.rows();
        if (nx_1>4){
            Eigen::VectorXd x(nx_1,1);
            x.setZero();
            
            Eigen::VectorXd q_v(4,1);
            q_v = r2q(x2.block(0,0,3,1));
            
            x.block(0,0,4,1) = quat_multiply(q_v, x1.block(0,0,4,1));
            x.block(4,0,nx_1-4,1) = x1.block(4,0,nx_1-4,1) + x2.block(3,0,nx_1-4,1);
            return x;
        }
        if (nx_1==4){
            Eigen::VectorXd x(nx_1,1);
            x.setZero();

            Eigen::VectorXd q_v(4,1);
            q_v = r2q(x2);
            x = quat_multiply(q_v, x1);
            return x;
        }
        if (nx_1<4){
            Eigen::VectorXd x(nx_1,1);
            x.setZero();
            x = x1 + x2;  
            return x;
        }
    }
    // Weighted mean operation
    static Eigen::VectorXd weighted_mean(const Eigen::MatrixXd &X, const Eigen::VectorXd &W){
        int nx = X.rows();
        Eigen::VectorXd x_mean(nx,1);
        int itr = 5; // Number of iterations
        int n_i = W.rows();
        
        if(nx>4){
            // Auxiliar variables
            Eigen::VectorXd x_q_inv(4,1);
            Eigen::VectorXd x_q_multiply(4,1);
            Eigen::VectorXd x_q_m(4,1);
            Eigen::VectorXd x_v_s(3,1);
            x_q_m = X.block(0,0,4,1);
        
            for(int j = 0; j<itr; j++){
            x_v_s.setZero();
            for(int i = 0; i<n_i; i++){
                //inverse of x_q_m
                x_q_inv(0,0) = x_q_m(0,0);
                x_q_inv.block(1,0,3,1) = -x_q_m.block(1,0,3,1);
                x_q_multiply = quat_multiply(X.block(0,i,4,1), x_q_inv);
                x_v_s += W(i,0)*q2r(x_q_multiply);
            }
            
            x_q_m = quat_multiply(r2q(x_v_s), x_q_m);
            }
            
            x_mean.block(0,0,4,1) = x_q_m;
            x_mean.block(4,0,nx-4,1) = X.block(4,0,nx-4,n_i)*W; 
        }
        if(nx==4){
            // Auxiliar variables
            Eigen::VectorXd x_q_inv(4,1);
            Eigen::VectorXd x_q_multiply(4,1);
            Eigen::VectorXd x_q_m(4,1);
            Eigen::VectorXd x_v_s(3,1);
            x_q_m = X.block(0,0,4,1);
        
            x_v_s.setZero();
            for(int j = 0; j<itr; j++){
            x_v_s.setZero();
            for(int i = 0; i<n_i; i++){
                //inverse of x_q_m
                x_q_inv(0,0) = x_q_m(0,0);
                x_q_inv.block(1,0,3,1) = -x_q_m.block(1,0,3,1);
                x_q_multiply = quat_multiply(X.block(0,i,4,1), x_q_inv);
                x_v_s += W(i,0)*q2r(x_q_multiply);
            }
        
            x_q_m = quat_multiply(r2q(x_v_s), x_q_m);
            }
            x_mean = x_q_m;
        }
        if(nx<4){
            x_mean = X*W;  
        }
        return x_mean;
    }
}
#endif // QUATERNIONTOOLS_h