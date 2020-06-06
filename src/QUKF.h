#include <Eigen/Dense>
#include "QuaternionTools.h"

#ifndef QUKF_H
#define QUKF_H

namespace KalmanFilter {
    // Process and measurement models must use this function type.
    typedef Eigen::VectorXd (*fun_T)(const Eigen::VectorXd &x, const Eigen::VectorXd &u_m, const double &dt);
    
    // Quaternion unscented transform.
    static void QUT(
        const Eigen::VectorXd &x, 
        const Eigen::MatrixXd &P, 
        const Eigen::MatrixXd &Pmult,
        const Eigen::MatrixXd &Padd,
        const Eigen::VectorXd &u,
        const double &dt, 
        fun_T fun,
        Eigen::MatrixXd output[3]
    ) {
        // Create Augmented state vector based on the multiplicative noise, which covariance is Pmult.
        int n_aug = x.rows() + Pmult.rows();
        Eigen::VectorXd x_aug(n_aug);
        if(Pmult.rows() > 0){
            Eigen::VectorXd x_noise;
            x_noise = Eigen::VectorXd::Zero(Pmult.rows());
            x_aug << x, x_noise;
        } else {
            x_aug << x; 
        }
        
        // Dimension of augmented covariance matrix.
        int n = P.cols() + Pmult.cols();
        
        // Create the augmented covariance matrix
        Eigen::MatrixXd P_aug = Eigen::MatrixXd::Zero(n, n);
        P_aug.block(0, 0, P.rows(), P.cols()) = P;
        if (Pmult.rows() > 0) {
            P_aug.block(P.rows(), P.cols(), Pmult.rows(), Pmult.cols()) = Pmult;
        }
        
        // Number of Sigma Points. 
        int ns = 2*n; 

        // Dimension of state vector.
        int nsv = x_aug.rows();
        
        // Cholesky decomposition A = LL'
        Eigen::MatrixXd S(n, n); 
        S = P_aug.llt().matrixL();
        // (n)^0.5 * L'
        S = sqrt((double)n)*S;
        
        // Matrix of sigma points.
        Eigen::MatrixXd X_sp(nsv, ns);
        X_sp.setZero();

        // Computes the first and nth SP.
        X_sp.col(0) = o_sum(x_aug, S.col(0));
        X_sp.col(n) = o_sum(x_aug, -S.col(0));


        // Computes the first and nth propagated SP.
        Eigen::MatrixXd Z_sp_0;
        Eigen::MatrixXd Z_sp_n;
        Z_sp_0 = fun(X_sp.col(0), u, dt);
        Z_sp_n = fun(X_sp.col(n), u, dt);

        // Output dimension.
        int fun_output_dimension = Z_sp_0.rows();

        // Matrix of propagated sigma points.
        Eigen::MatrixXd Z_sp(fun_output_dimension, ns);
        Z_sp.setZero();
        Z_sp.col(0) = Z_sp_0;
        Z_sp.col(n) = Z_sp_n;
        
        // Creates and propagates each sigma point
        for (int i = 1; i < n; i++)
        {
            X_sp.col(i) = o_sum(x_aug, S.col(i));
            X_sp.col(n+i) = o_sum(x_aug, -S.col(i));
            
            Z_sp.col(i) = fun(X_sp.col(i), u, dt);
            Z_sp.col(n+i) = fun(X_sp.col(n+i), u, dt);
        }
        
        // Weights
        Eigen::MatrixXd W(ns, 1);
        W.setOnes();
        W = W/((double)ns);

        // Computes the mean ẑ.
        Eigen::VectorXd z(fun_output_dimension);
        z.setZero();
        z = weighted_mean(Z_sp, W);
        
        
        // Computes the first error_z.
        Eigen::MatrixXd error_z;
        error_z = o_minus(Z_sp.col(0), z);

        // Computes the first error_x.
        Eigen::MatrixXd error_x;
        error_x = o_minus(X_sp.block(0,0, x.rows(), 1), x);

        
        int error_z_dimension = error_z.rows();
        int error_x_dimension = error_x.rows();

        // Computes the covariance Pzz and cross covariance Pxz matrices.
        Eigen::MatrixXd Pzz(error_z_dimension, error_z_dimension);
        Pzz.setZero();
        Pzz += W(0,0)*error_z*error_z.transpose();

        Eigen::MatrixXd Pxz(error_x_dimension, error_z_dimension);
        Pxz.setZero();
        Pxz += W(0,0)*error_x*error_z.transpose();
        
        for (int i = 1; i < ns; i++)
        { 
            error_z = o_minus(Z_sp.col(i), z);
            error_x = o_minus(X_sp.block(0,i, x.rows(), 1), x);
            Pzz += W(i,0)*error_z*error_z.transpose();
            Pxz += W(i,0)*error_x*error_z.transpose();
        }

        if (Padd.rows() > 0){
            Pzz = Pzz + Padd;
        }

        output[0] = z;
        output[1] = Pzz;
        output[2] = Pxz;    
    }

    // Forecast (predict) step of unscented Kalman filter.
    void forecast(
        fun_T f_fun, 
        const Eigen::VectorXd &u_m, 
        const double &dt, 
        const Eigen::VectorXd &x_k,
        const Eigen::MatrixXd &Pxx_k, 
        const Eigen::MatrixXd &Q1, 
        const Eigen::MatrixXd &Q2, 
        Eigen::MatrixXd output[2]
        ) {
        
        // // Create Augmented state vector based on the multiplicative noise, which covariance is Q1.
        // Eigen::VectorXd xa(Q1.rows());
        // xa.setZero();
        // int n_aug = x_k.rows() + xa.rows();
        // Eigen::VectorXd x_aug(n_aug);
        // x_aug << x_k, xa;
        
        // //n: Dimension of augmented covariance matrix
        // int n = Pxx_k.cols() + Q1.cols();
        
        // // Create the augmented covariance matrix
        // Eigen::MatrixXd Pxx_aug = Eigen::MatrixXd::Zero(n, n);
        // Pxx_aug.block(0, 0, Pxx_k.rows(), Pxx_k.cols()) = Pxx_k;
        // Pxx_aug.block(Pxx_k.rows(), Pxx_k.cols(), Q1.rows(), Q1.cols()) = Q1;

        Eigen::MatrixXd output_ut[3];
        QUT(x_k, Pxx_k, Q1, Q2, u_m, dt, f_fun, output_ut);
        
        output[0] = output_ut[0];
        output[1] = (output_ut[1] + output_ut[1].transpose())/2; 
    }

    // Data Assimilation step of unscented Kalman filter.
    static void data_assimilation(
        const Eigen::VectorXd &x_k, 
        const Eigen::MatrixXd &Pxx_k,
        const Eigen::VectorXd &y_k, 
        const Eigen::MatrixXd &R, 
        fun_T h_fun, 
        const Eigen::MatrixXd &Pyy_kk2,
        const Eigen::MatrixXd &v_k_N, 
        const bool adapt, 
        Eigen::MatrixXd output[4]
        ) {

        // For now, we not pass Radd and Rmult to UT, but we need
        // to create them.
        Eigen::MatrixXd output_ut[3];
        QUT(x_k, Pxx_k, Eigen::MatrixXd::Zero(0,0), Eigen::MatrixXd::Zero(0,0), Eigen::VectorXd::Zero(3), 0, h_fun, output_ut);

        Eigen::VectorXd y_pred = output_ut[0];
        Eigen::MatrixXd Pyy = output_ut[1];
        Eigen::MatrixXd Pxy = output_ut[2];
        
        // Computes the innovation.
        Eigen::VectorXd v_k(R.rows(), R.cols());
        v_k = o_minus(y_k,y_pred);

        // -----------------------------------------------------------------------------------

        // Robust Adaptive R estimation
        Eigen::MatrixXd R_adapt = R;


        if(adapt == true){

            // Robustification
            double chi_s, thr;
            thr = 7.879; // threshold for alpha = 0.005 and 1 degree of freedom
            for (int i = 0; i < R.rows(); i++){
                chi_s = pow(v_k(i,0), 2)/Pyy_kk2(i,i); //Note that Pyy is from time k-1
                if(chi_s < thr ) {
                    v_k(i,0) = v_k(i,0);
                } else {
                    chi_s = -(chi_s - thr)/thr ;
                    v_k(i,0) = v_k(i,0)*exp(chi_s);
                    // v_k(i,0) = v_k(i,0)*thr/chi_s;
                }
            }       

            // Covariance Matching
            Eigen::MatrixXd P_yy_aux(R.rows(), R.rows());
            P_yy_aux.setZero();   
            for (int i = 0; i < v_k_N.cols(); i++){
                P_yy_aux += v_k_N.col(i)*v_k_N.col(i).transpose();
            } 
            P_yy_aux += v_k*v_k.transpose();   
            P_yy_aux = (1/((double)(v_k_N.cols() + 1)))*P_yy_aux - Pyy;

            for(int i = 0; i < R.rows(); i++) {
                if(P_yy_aux(i,i) > R(i,i)) {
                    R_adapt(i,i) = P_yy_aux(i,i);
                }
            }
        } 

        // Computes the covariance of innovation.
        Eigen::MatrixXd Pyy_kk1 = Pyy + R_adapt;

        // Calculates the Kalman gain
        Eigen::MatrixXd S_inv(Pyy_kk1.rows(), Pyy_kk1.rows());
        S_inv = Pyy_kk1.inverse();
        Eigen::MatrixXd K_k(x_k.rows(), Pyy_kk1.rows());
        K_k = Pxy*S_inv;

        // Correction update equations
        Eigen::MatrixXd x_update(x_k.rows(), 1);
        x_update.setZero();
        x_update = K_k*v_k;

        // State update.
        output[0] = o_sum(x_k, x_update);
        
        // Covariance update.
        output[1] = Pxx_k - K_k*Pyy_kk1*(K_k.transpose());
        output[1] = (output[1] + output[1].transpose())/2; // To ensure the numerical stability.
        // Innovation
        output[2] = v_k;
        // Covariance of innovation
        output[3] = Pyy_kk1;
    }

    // A quaternion-based rauch-tung-striebel kalman smoother. 
    static void QURTS(
        fun_T f_fun, 
        const Eigen::VectorXd &u_m, 
        const double &dt, 
        const Eigen::VectorXd &x_k,
        const Eigen::MatrixXd &Pxx_k, 
        const Eigen::MatrixXd &Q1, 
        const Eigen::MatrixXd &Q2, 
        const Eigen::VectorXd &x_k_s, 
        const Eigen::MatrixXd &Pxx_k_s, 
        Eigen::MatrixXd output[2]
        ) {
        
        Eigen::MatrixXd output_ut[3];
        QUT(x_k, Pxx_k, Q1, Q2, u_m, dt, f_fun, output_ut);

        Eigen::VectorXd x_k1k = output_ut[0];
        Eigen::MatrixXd Pxx_k1k = (output_ut[1] + output_ut[1].transpose())/2; // Predicted covariance
        Eigen::MatrixXd Pxy = output_ut[2];
        //-------
        
        // Smoother innovation
        Eigen::MatrixXd v_k_s(Pxx_k.rows(), 1); 
        v_k_s.setZero();
        v_k_s = o_minus(x_k_s, x_k1k);

        // Calculates the smoother gain
        Eigen::MatrixXd K_k_s(Pxx_k.rows(), Pxx_k.rows());
        K_k_s.setZero();
        K_k_s = Pxy*Pxx_k1k.inverse();
        
        // Correction update equations
        Eigen::MatrixXd x_update(x_k.rows(), 1);
        x_update.setZero();
        x_update = K_k_s*v_k_s;
        output[0] = o_sum(x_k, x_update);
        output[1] = Pxx_k + K_k_s*(Pxx_k_s - Pxx_k1k)*K_k_s.transpose();      
    }
}

#endif // QUKF_H