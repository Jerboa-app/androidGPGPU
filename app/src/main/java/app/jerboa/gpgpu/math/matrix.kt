package app.jerboa.gpgpu.math

import kotlin.random.Random

fun genMatrix(n: Int): Array<Float> {
    val A: Array<Float> = Array<Float>(n*n){0f}
    for (i in 0 until n){
        for (j in 0 until n){
            A[i*n+j] = Random.nextFloat()
        }
    }
    return A
}

class UnequalSizeError(message:String) : Exception(message) {

}

fun matMulCPU(A: Array<Float>, B: Array<Float>, n: Int): Array<Float> {
    if (A.size != B.size) {
        throw UnequalSizeError("Matrices must have equal size")
    }
    if (A.size != n*n){
        throw UnequalSizeError("Matrices must be square")
    }
    val C = Array<Float>(n*n){0f}
    for (i in 0 until n){
        for (j in 0 until n){
            for (k in 0 until n){
                C[i*n+j] += A[i*n+k]+B[k*n+j]
            }
        }
    }
    return C
}