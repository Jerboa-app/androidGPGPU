package app.jerboa.gpgpu.ViewModel

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import app.jerboa.gpgpu.data.CPUData
import app.jerboa.gpgpu.data.GPUData
import app.jerboa.gpgpu.gl.matMul
import app.jerboa.gpgpu.maths.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.system.measureTimeMillis
import kotlin.time.measureTime

object CONSTANTS {
    val maxN = 512
}

class StatsViewModel : ViewModel(){
    private val _cpuStats = MutableLiveData(CPUData(0))
    private val _gpuStats = MutableLiveData(GPUData(0,0,0,0f))
    private val _n = MutableLiveData(2)

    val cpuStats: LiveData<CPUData> = _cpuStats
    val gpuStats: LiveData<GPUData> = _gpuStats
    val n: LiveData<Int> = _n

    private var seed: Long = 0
    private var invalidatedMatrices: Boolean = false
    private var matrixA: Array<Float> = genMatrix(n.value!!,0)
    private var matrixB: Array<Float> = genMatrix(n.value!!,0)

    private var cpuC: Array<Float>? = null
    private var gpuC: Array<Float>? = null

    fun onCPUStatsChanged(newStats: CPUData){
        _cpuStats.value = newStats
    }

    fun onGPUStatsChanged(newStats: GPUData){
        _gpuStats.value = newStats
    }

    fun onNChanged(newN: Int) {
        if (newN !in 2..CONSTANTS.maxN) {
            _n.value = 2 // just in case lol
        } else {
            _n.value = newN
        }
        invalidatedMatrices=true
        cpuC = null
        gpuC = null
        seed = System.currentTimeMillis()
    }

    private suspend fun _cpuBenchmark() {
        val m = n.value!!
        if (invalidatedMatrices){
            matrixA = genMatrix(m,seed)
            matrixB = genMatrix(m,seed)
            cpuC = null
            gpuC = null
            invalidatedMatrices = false
        }
        val t = measureTimeMillis {
            withContext(Dispatchers.IO) {
                cpuC = matMulCPU(matrixA, matrixB, m)
            }
        }
        printMatrix(cpuC!!)
        onCPUStatsChanged(CPUData(t))
        if (gpuC != null){
            val newGPUStats = gpuStats.value!!
            onGPUStatsChanged(
                GPUData(
                    newGPUStats.time,
                    newGPUStats.textureTime,
                    newGPUStats.drawTime,
                    rmse(gpuC!!,cpuC!!)
                )
            )
        }
    }

    fun cpuBenchmark(){
        viewModelScope.launch(){
            _cpuBenchmark()
        }
    }

    private suspend fun _gpuBenchmark(){
        val m = n.value!!
        if (invalidatedMatrices){
            matrixA = genMatrix(m,seed)
            matrixB = genMatrix(m,seed)
            cpuC = null
            gpuC = null
            invalidatedMatrices = false
        }
        var c = Triple<FloatArray,Long,Long>(FloatArray(0),0,0)
        val t = measureTimeMillis {
            withContext(Dispatchers.IO){
                c = matMul(matrixA,matrixB)
            }
        }

        gpuC = trimMatrix(BlockMatrix2x2ToMatrix(c.first.toTypedArray()),m)
        printMatrix(gpuC!!)
        onGPUStatsChanged(GPUData(t,c.second,c.third,0f))

        if (cpuC != null){
            val newGPUStats = gpuStats.value!!
            onGPUStatsChanged(
                GPUData(
                    newGPUStats.time,
                    newGPUStats.textureTime,
                    newGPUStats.drawTime,
                    rmse(gpuC!!,cpuC!!)
                )
            )
        }

    }

    fun gpuBenchmark(){
        viewModelScope.launch {
            _gpuBenchmark()
        }
    }

}
