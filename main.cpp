#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace std;
using namespace Eigen;

double rel_error(const VectorXd& vec1, const VectorXd& vec2)
{
    return (vec1-vec2).norm()/vec2.norm();
}

int main()
{
    VectorXd x(2);
    x << -1.0, -1.0;

    // problema 1

    MatrixXd A1(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01,
    -9.992887623566787e-01;

    VectorXd b1(2);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    
    FullPivLU<MatrixXd> LU1(A1);
    VectorXd x1_lu = LU1.solve(b1);

    HouseholderQR<MatrixXd> QR1(A1);
    VectorXd x1_qr = QR1.solve(b1);

    cout << "Problem 1:" << endl;
    cout << scientific << setprecision(16) << "PALU solution:\n" << x1_lu << endl;
    cout << "PALU error: " << rel_error(x1_lu, x) << endl;
    cout << scientific << setprecision(16) << "QR solution:\n" << x1_qr << endl;
    cout << "QR error: " << rel_error(x1_qr, x) << endl;
    cout << "--- \n" << endl;

    // problema 2

    MatrixXd A2(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01,
    -8.324762492991313e-01;

    VectorXd b2(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    
    FullPivLU<MatrixXd> LU2(A2);
    VectorXd x2_lu = LU2.solve(b2);

    HouseholderQR<MatrixXd> QR2(A2);
    VectorXd x2_qr = QR2.solve(b2);

    cout << "Problem 2:" << endl;
    cout << scientific << setprecision(16) << "PALU solution:\n" << x2_lu << endl;
    cout << "PALU error: " << rel_error(x2_lu, x) << endl;
    cout << scientific << setprecision(16) << "QR solution:\n" << x2_qr << endl;
    cout << "QR error: " << rel_error(x2_qr, x) << endl;
    cout << "--- \n" << endl;

    // problema 3

    MatrixXd A3(2, 2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01,
    -8.320502947645361e-01;

    VectorXd b3(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    
    FullPivLU<MatrixXd> LU3(A3);
    VectorXd x3_lu = LU3.solve(b3);

    HouseholderQR<MatrixXd> QR3(A3);
    VectorXd x3_qr = QR3.solve(b3);

    cout << "Problem 3:" << endl;
    cout << scientific << setprecision(16) << "PALU solution:\n" << x3_lu << endl;
    cout << "PALU error: " << rel_error(x3_lu, x) << endl;
    cout << scientific << setprecision(16) << "QR solution:\n" << x3_qr << endl;
    cout << "QR error: " << rel_error(x3_qr, x) << endl;
    
    return 0;
}
