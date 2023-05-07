#include <cupy/complex.cuh>

#define get_x(_coord, _X) \
  (((_coord[3] * _X[2] + _coord[2]) * _X[1] + _coord[1]) * _X[0] + _coord[0])

template <typename T, int Nc>
class Matrix {
private:
  T data[Nc][Nc];

public:
  __device__ __host__ Matrix()
  {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        data[i][j] = 0;
      }
    }
  }

  __device__ __host__ Matrix(const T *source)
  {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        data[i][j] = source[i * Nc + j];
      }
    }
  }

  __device__ __host__ Matrix(const Matrix<T, Nc> &matrix)
  {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        data[i][j] = matrix[i][j];
      }
    }
  }

  __device__ __host__ const T *operator[](const int i) const
  {
    return data[i];
  }

  __device__ __host__ T *operator[](const int i)
  {
    return data[i];
  }

  __device__ __host__ void operator=(const Matrix<T, Nc> &rhs)
  {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        data[i][j] = rhs[i][j];
      }
    }
  }

  __device__ __host__ Matrix<T, Nc> operator+(const T &rhs)
  {
    Matrix<T, Nc> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = data[i][j];
      }
      result[i][i] += rhs;
    }
    return result;
  }

  __device__ __host__ Matrix<T, Nc> operator+(const Matrix<T, Nc> &rhs)
  {
    Matrix<T, Nc> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = data[i][j] + rhs[i][j];
      }
    }
    return result;
  }

  __device__ __host__ void operator+=(const Matrix<T, Nc> &rhs)
  {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        data[i][j] += rhs[i][j];
      }
    }
  }

  __device__ __host__ void operator-=(const T &rhs)
  {
    for (int i = 0; i < Nc; ++i) {
      data[i][i] -= rhs;
    }
  }

  __device__ __host__ Matrix<T, Nc> operator*(const T &rhs)
  {
    Matrix<T, Nc> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = data[i][j] * rhs;
      }
    }
    return result;
  }

  __device__ __host__ Matrix<T, Nc> operator*(const Matrix<T, Nc> &rhs)
  {
    Matrix<T, Nc> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = 0;
        for (int k = 0; k < Nc; ++k) {
          result[i][j] += data[i][k] * rhs[k][j];
        }
      }
    }
    return result;
  }
};

template <typename T, int Nc>
__device__ __host__ T trace(const Matrix<T, Nc> &matrix)
{
  T result = 0;
  for (int i = 0; i < Nc; ++i) {
    result += matrix[i][i];
  }
  return result;
}

template <typename T, int Nc>
__device__ __host__ Matrix<T, Nc> adjoint(const Matrix<T, Nc> &matrix)
{
  Matrix<T, Nc> result;
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      result[i][j] = conj(matrix[j][i]);
    }
  }
  return result;
}

template <typename T, int Nc>
__device__ __host__ Matrix<T, Nc> antiherm(const Matrix<T, Nc> &matrix)
{
  Matrix<T, Nc> result;
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      result[i][j].real((matrix[j][i].imag() + matrix[i][j].imag()) / 2.);
      result[i][j].imag((matrix[j][i].real() - matrix[i][j].real()) / 2.);
    }
  }
  result -= trace(result) / T(Nc);
  return result;
}

template <typename T>
__global__ void stout_smear(complex<T> *U_out, const complex<T> *U_in, const T rho, const int Lx, const int Ly, const int Lz, const int Lt)
{
  const int Nd = 4;
  const int Nc = 3;
  typedef Matrix<complex<T>, Nc> ColorMatrix;

  const int volume = Lx * Ly * Lz * Lt;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int mu = blockIdx.y * blockDim.y + threadIdx.y;
  if (mu >= Nd - 1 || x >= volume) {
    return;
  }
  const int X[Nd] = {Lx, Ly, Lz, Lt};
  int coord[Nd] = {x % Lx, x / Lx % Ly, x / (Lx * Ly) % Lz, x / (Lx * Ly * Lz) % Lt};

  ColorMatrix U(U_in + (mu * volume + x) * Nc * Nc);
  ColorMatrix Q;
  for (int nu = 0; nu < Nd - 1; ++nu) {
    if (nu != mu) {
      ColorMatrix staple1(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      staple1 = staple1 * ColorMatrix(U_in + (mu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      staple1 = staple1 * adjoint(ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc));
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      ColorMatrix staple2(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      staple2 = adjoint(staple2) * ColorMatrix(U_in + (mu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      staple2 = staple2 * ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      Q += staple1 + staple2;
    }
  }
  Q = antiherm((Q * rho) * adjoint(U));

  ColorMatrix Q_sq = Q * Q;
  double c0 = trace(Q_sq * Q).real() / 3;
  double c1 = trace(Q_sq).real() / 2;
  double c0_max = 2 * c1 / 3 * sqrt(c1 / 3);
  int parity = 0;
  if (c0 < 0) {
    parity = 1;
    c0 *= -1;
  }
  double theta = acos(c0 / c0_max);
  double u = sqrt(c1 / 3) * cos(theta / 3);
  double w = sqrt(c1) * sin(theta / 3);
  double u_sq = u * u;
  double w_sq = w * w;
  double e_iu_real = cos(u);
  double e_iu_imag = sin(u);
  double e_2iu_real = cos(2 * u);
  double e_2iu_imag = sin(2 * u);
  double cos_w = cos(w);
  double sinc_w = 1 - w_sq / 6 * (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)));
  if (abs(w) > 0.05) {
    sinc_w = sin(w) / w;
  }
  double f_denom = 1 / (9 * u_sq - w_sq);
  double f0_real = ((u_sq - w_sq) * e_2iu_real + e_iu_real * 8 * u_sq * cos_w + e_iu_imag * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f0_imag = ((u_sq - w_sq) * e_2iu_imag - e_iu_imag * 8 * u_sq * cos_w + e_iu_real * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f1_real = (2 * u * e_2iu_real - e_iu_real * 2 * u * cos_w + e_iu_imag * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f1_imag = (2 * u * e_2iu_imag + e_iu_imag * 2 * u * cos_w + e_iu_real * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f2_real = (e_2iu_real - e_iu_real * cos_w - e_iu_imag * 3 * u * sinc_w) * f_denom;
  double f2_imag = (e_2iu_imag + e_iu_imag * cos_w - e_iu_real * 3 * u * sinc_w) * f_denom;
  if (parity) {
    f0_imag *= -1;
    f1_real *= -1;
    f2_imag *= -1;
  }
  complex<T> f0 = {T(f0_real), T(f0_imag)};
  complex<T> f1 = {T(f1_real), T(f1_imag)};
  complex<T> f2 = {T(f2_real), T(f2_imag)};
  ColorMatrix e_iQ = Q_sq * f2 + Q * f1 + f0;
  U = e_iQ * U;
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      U_out[(mu * volume + x) * Nc * Nc + i * Nc + j] = U[i][j];
    }
  }
}
