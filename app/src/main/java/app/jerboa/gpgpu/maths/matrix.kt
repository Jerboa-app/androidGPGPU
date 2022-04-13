package app.jerboa.gpgpu.maths

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

// random square n by n matrices
fun genMatrix(n: Int, seed: Long = 0): Array<Float> {
    val A: Array<Float> = Array<Float>(n*n){0f}
    for (i in 0 until n){
        for (j in 0 until n){
            A[i*n+j] = Random(seed+i.toLong()+seed*j.toLong()).nextFloat()
        }
    }
    return A
}

class UnequalSizeError(message:String) : Exception(message) {

}

// naive cpu multiplication, can get optimise this if you want!
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
                C[i*n+j] += A[i*n+k]*B[k*n+j]
            }
        }
    }
    return C
}

// convert to a block matrix compose of 2 by 2 blocks
fun matrixTo2x2BlockMatrix(A: Array<Float>): Array<Float> {
    var n: Int = sqrt(A.size.toDouble()).toInt()
    val Ba: Array<Float>
    // if not divisible by 4 make it so!
    if (n%4 != 0){
        val m = n + (4 - n % 4) // raise to next multiple of 4
        Ba = Array<Float>(m*m){0f}
        for (i in 0 until n){
            for (j in 0 until n){
                Ba[i*m+j] = A[i*n+j]
            }
        }
        n = m
    }
    else{
        Ba = A
    }
    val nb = ceil(sqrt(n*n/4f)).toInt() // size of block matrix
    val T = Array<Float>(4*nb*nb){0f} // the block matrix, in flat form (row major)
    for (i in 1 until n+1){
        for (j in 1 until n+1){
            val bi: Int = ceil(i/2f).toInt()
            val bj: Int = ceil(j/2f).toInt()
            val li = floor(i/bi.toFloat()).toInt()
            val lj = floor(j/bj.toFloat()).toInt()
            // blockIdx.x*n + blockIdx.y + local index in block
            T[4*( (bi-1)*nb+bj-1)+2*(li-1)+lj-1] = Ba[(i-1)*n+j-1]
        }
    }
    return T
}

// convert back from block matrix
fun BlockMatrix2x2ToMatrix(A: Array<Float>): Array<Float>{
    val nb = ceil(sqrt(A.size/4.0f)).toInt()
    val n = ceil(sqrt(4f*nb*nb)).toInt()
    val T = Array<Float>(n*n){0f}
    for (i in 1 until n+1){
        for (j in 1 until n+1){
            val bi: Int = ceil(i/2f).toInt()
            val bj: Int = ceil(j/2f).toInt()
            val li = floor(i/bi.toFloat()).toInt()
            val lj = floor(j/bj.toFloat()).toInt()
            // blockIdx.x*n + blockIdx.y + local index
            T[(i-1)*n+j-1] = A[4*( (bi-1)*nb+bj-1)+2*(li-1)+lj-1]
        }
    }
    return T
}

// trim out extra bits added if the original matrix was not divisible by 4
fun trimMatrix(A: Array<Float>, n: Int): Array<Float>{
    val T = Array<Float>(n*n){0f}
    val m = ceil(sqrt(A.size.toDouble())).toInt()
    var k = 0
    for (i in 0 until m){
        for (j in 0 until m){
            if (i < n && j < n){
                T[k] = A[i*m+j]
                k+=1
            }
            if (k >= n*n){
                return T
            }
        }
    }
    return T
}

// root-mean square error
fun rmse(A:Array<Float>,B:Array<Float>): Float {
    var e = 0f
    for (i in A.indices){
        e += (A[i] - B[i]).pow(2)
    }
    return sqrt(e / A.size.toFloat())
}

fun printMatrix(A:Array<Float>){
    val n = ceil(sqrt(A.size.toDouble())).toInt()
    for (i in 0 until n){
        for (j in 0 until n){
            print(A[i*n+j].toString()+", ")
        }
        println()
    }

}