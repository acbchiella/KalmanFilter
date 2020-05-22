#include <Eigen/Dense>
#include "QuaternionTools.h"

#ifndef QUKF_H
#define QUKF_H

namespace KalmanFilter {
    // Process model function type
    typedef Eigen::VectorXd (*f_fun_T)(const Eigen::VectorXd &x, const Eigen::VectorXd &u_m, const double &dt);
    //Measurement model
    typedef Eigen::VectorXd (*h_fun_T)(const Eigen::VectorXd &x);

    // Forecast (predict) step of unscented Kalman filter
    void forecast(f_fun_T f_fun, const Eigen::VectorXd &u_m, const double &dt, const Eigen::VectorXd &x_k,
        const Eigen::MatrixXd &Pxx_k, const Eigen::MatrixXd &Q1, const Eigen::MatrixXd &Q2, Eigen::MatrixXd output[2]){
        
        // Create Augmented state vector based on the multiplicative noise, which covariance is Q1.
        
        Eigen::VectorXd xa(Q1.rows());
        xa.setZero();
        int n_aug = x_k.rows() + xa.rows();
        Eigen::VectorXd x_aug(n_aug);
        x_aug << x_k, xa;
        
        //n: Dimension of augmented covariance matrix
        int n = Pxx_k.cols() + Q1.cols();
        
        // Create the augmented covariance matrix
        Eigen::MatrixXd Pxx_aug = Eigen::MatrixXd::Zero(n, n);
        Pxx_aug.block(0, 0, Pxx_k.rows(), Pxx_k.cols()) = Pxx_k;
        Pxx_aug.block(Pxx_k.rows(), Pxx_k.cols(), Q1.rows(), Q1.cols()) = Q1;
        
        int ns = 2*n; // Number of Sigma Points
        
        Eigen::MatrixXd S(n, n); // Cholesky decomposition A = LL'
        // S = Pxx_aug.transpose().llt().matrixL();
        S = Pxx_aug.llt().matrixL();
        S = sqrt((double)n)*S;
        
        // Initializes the matrix of sigma points.
        Eigen::MatrixXd X_sp(n_aug, ns);
        X_sp.setZero();
        // Initializes the matrix of propagate the Sigma Points
        Eigen::MatrixXd Y_sp(x_k.rows(), ns);
        Y_sp.setZero();
        
        // Creates and propagates each sigma point
        for (int i = 0; i < n; i++)
        {
            X_sp.col(i) = o_sum(x_aug, S.col(i));
            X_sp.col(n+i) = o_sum(x_aug, -S.col(i));
            
            Y_sp.col(i) = f_fun(X_sp.col(i), u_m, dt);
            Y_sp.col(n+i) = f_fun(X_sp.col(n+i), u_m, dt);
        }
        
        // Computes the predicted.
        Eigen::MatrixXd W(ns, 1);
        W.setOnes();
        W = W/((double)ns);
        Eigen::VectorXd x_kk1(x_k.rows());
        x_kk1.setZero();
        x_kk1 = weighted_mean(Y_sp, W);
        
        
        // Computes the predicted covariance matrix.
        Eigen::MatrixXd Pxx_kk1(Pxx_k.rows(),Pxx_k.cols());
        Pxx_kk1.setZero();
        Eigen::MatrixXd erro(Pxx_k.rows(), 1);
        erro.setZero();
        for (int i = 0; i < ns; i++)
        { 
            erro = o_minus(Y_sp.col(i), x_kk1);
            Pxx_kk1 += W(i,0)*erro*erro.transpose();
        }
                
        output[0] = x_kk1;
        output[1] = ((Pxx_kk1 + Q2) + (Pxx_kk1 + Q2).transpose())/2; 
    }

    // Data Assimilation step
   static void data_assimilation(const Eigen::VectorXd &x_k, const Eigen::MatrixXd &Pxx_k,
        const Eigen::VectorXd &y_k, const Eigen::MatrixXd &R, h_fun_T h_fun, const Eigen::MatrixXd &Pyy_kk2,
        const Eigen::MatrixXd &v_k_N, const bool adapt, Eigen::MatrixXd output[4]) {
    
        // n: Number of elements in the state vector 
        int n_x = x_k.rows();
        // n: Number of elements in the state vector -1
        int n = Pxx_k.rows();
        // Compute the UT
        int ns = 2*n; // Number of Sigma Points

        // Calculate weights
        Eigen::MatrixXd W(ns, 1);
        W.setOnes();
        W = W/(((double)ns));

        Eigen::MatrixXd S(n, n); // Cholesky decomposition A = LL'
        // S = Pxx_k.transpose().llt().matrixL();
        S = Pxx_k.llt().matrixL();
        S = sqrt((double)n)*(S);
        
        // Initialize the matrix of sigma points
        Eigen::MatrixXd X_sp(n_x, ns);
        X_sp.setZero();
        // Initialize the matrix of propagate the Sigma Points
        Eigen::MatrixXd Y_sp(y_k.rows(), ns);
        // Create each sigma point
        for (int i = 0; i < n; i++)
        {
            X_sp.col(i) = o_sum(x_k, S.col(i));
            X_sp.col(n+i) = o_sum(x_k, -S.col(i));
            // Propagate the Sigma Points
            Y_sp.col(i) = h_fun(X_sp.col(i));
            Y_sp.col(n+i) = h_fun(X_sp.col(n+i));
        }

        // Compute the prediction of measurement
        Eigen::VectorXd y_pred(y_k.rows());
        y_pred.setZero();
        y_pred = weighted_mean(Y_sp, W);

        // Covariance matrix of innovation
        Eigen::MatrixXd Pyy(R.rows(), R.rows());
        Pyy.setZero();
        // Error y
        Eigen::MatrixXd erro_y(R.rows(), 1);
        erro_y.setZero();
        // Error x
        Eigen::MatrixXd erro_x(Pxx_k.rows(), 1);
        erro_x.setZero();
        // Cross covariance matrix
        Eigen::MatrixXd Pxy(Pxx_k.rows(), R.rows());
        Pxy.setZero();
        
        for (int i = 0; i < ns; i++)
        { 
            erro_y = o_minus(Y_sp.col(i),  y_pred);
            erro_x = o_minus(X_sp.col(i),  x_k);
            Pxy += W(i,0)*erro_x*erro_y.transpose();
            Pyy += W(i,0)*erro_y*erro_y.transpose();
        }
        
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

    // QURTS 
   static void QURTS(f_fun_T f_fun, const Eigen::VectorXd &u_m, const double &dt, const Eigen::VectorXd &x_k,
        const Eigen::MatrixXd &Pxx_k, const Eigen::MatrixXd &Q1, const Eigen::MatrixXd &Q2, 
        const Eigen::VectorXd &x_k_s, const Eigen::MatrixXd &Pxx_k_s, Eigen::MatrixXd output[2]){
        
       // Prediciton step
       // Create Augmented state vector based on the Q1
        // The state vector is augmented with the multiplicative noise
        Eigen::VectorXd xa(Q1.rows());
        xa.setZero();
        int n_aug = x_k.rows()+xa.rows();
        Eigen::VectorXd x_aug(n_aug);
        x_aug << x_k, xa;
        
        //n: Dimension of augmented covariance matrix
        int n = Pxx_k.cols()+Q1.cols();

        // Create the Augmented covariance matrix
        Eigen::MatrixXd Pxx_aug = Eigen::MatrixXd::Zero(n, n);
        Pxx_aug.block(0,0,Pxx_k.rows(),Pxx_k.cols()) = Pxx_k;
        Pxx_aug.block(Pxx_k.rows(), Pxx_k.cols(), Q1.rows(), Q1.cols()) = Q1;
        
        // Compute the UT
        int ns = 2*n; // Number of Sigma Points
        
        Eigen::MatrixXd S; // Cholesky decomposition A = LL'
        // S = Pxx_aug.transpose().llt().matrixL();
        S = Pxx_aug.llt().matrixL();
        S = sqrt((double)n)*(S);
        
        // Initialize the matrix of sigma points
        Eigen::MatrixXd X_sp(n_aug, ns);
        X_sp.setZero();
        // Initialize the matrix of propagate the Sigma Points
        Eigen::MatrixXd Y_sp(x_k.rows(), ns);
        Y_sp.setZero();
        
        // Create each sigma point
        for (int i = 0; i < n; i++)
        {
            X_sp.col(i) = o_sum(x_aug, S.col(i));
            X_sp.col(n+i) = o_sum(x_aug, -S.col(i));
            
            Y_sp.col(i) = f_fun(X_sp.col(i), u_m, dt);
            Y_sp.col(n+i) = f_fun(X_sp.col(n+i), u_m, dt);
        }
        
        // Compute the prediction of mean
        Eigen::MatrixXd W(ns, 1);
        W.setOnes();
        W = W/((double)ns);
        Eigen::VectorXd x_kk1(x_k.rows());
        x_kk1.setZero();
        x_kk1 = weighted_mean(Y_sp, W); // Predicted mean
        
        // Compute the prediction of covariance matrix
        Eigen::MatrixXd Pxx_kk1(Pxx_k.rows(),Pxx_k.cols());
        Pxx_kk1.setZero();
        Eigen::MatrixXd erro_x(Pxx_k.rows(), 1);
        erro_x.setZero();
        Eigen::MatrixXd erro_y(Pxx_k.rows(), 1);
        erro_y.setZero();
        // Cross covariance matrix
        Eigen::MatrixXd Pxy(Pxx_k.rows(), Pxx_k.rows());
        Pxy.setZero();
        
        for (int i = 0; i < ns; i++)
        { 
            erro_x = o_minus(Y_sp.col(i), x_kk1);
            erro_y = o_minus(Y_sp.col(i),  x_kk1);
            Pxx_kk1 += W(i,0)*erro_x*erro_x.transpose();
            Pxy += W(i,0)*erro_x*erro_y.transpose();
        }
                
        Pxx_kk1 = ((Pxx_kk1 + Q2) + (Pxx_kk1 + Q2).transpose())/2; // Predicted covariance
        
        // Smoother innovation
        Eigen::MatrixXd v_k_s(Pxx_k.rows(), 1); 
        v_k_s.setZero();
        v_k_s = o_minus(x_k_s, x_kk1);

        // Calculates the smoother gain
        Eigen::MatrixXd K_k_s(Pxx_k.rows(), Pxx_k.rows());
        K_k_s.setZero();
        K_k_s = Pxy*Pxx_kk1.inverse();
        
        // Correction update equations
        Eigen::MatrixXd x_update(x_k.rows(), 1);
        x_update.setZero();
        x_update = K_k_s*v_k_s;
        output[0] = o_sum(x_k, x_update);
        output[1] = Pxx_k + K_k_s*(Pxx_k_s - Pxx_kk1)*K_k_s.transpose();      
    }
}

#endif // QUKF_H