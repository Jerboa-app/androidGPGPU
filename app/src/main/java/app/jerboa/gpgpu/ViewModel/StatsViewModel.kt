package app.jerboa.gpgpu.ViewModel

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import app.jerboa.gpgpu.data.CPUData
import app.jerboa.gpgpu.data.GPUData
import app.jerboa.gpgpu.math.genMatrix
import app.jerboa.gpgpu.math.matMulCPU
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.system.measureTimeMillis

object CONSTANTS {
    val maxN = 1024
}

class StatsViewModel : ViewModel(){
    private val _cpuStats = MutableLiveData(CPUData(0))
    private val _gpuStats = MutableLiveData(GPUData(0,0,0,0))
    private val _n = MutableLiveData(2)

    val cpuStats: LiveData<CPUData> = _cpuStats
    val gpuStats: LiveData<GPUData> = _gpuStats
    val n: LiveData<Int> = _n

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
    }

    private suspend fun _cpuBenchmark() {
        val t = measureTimeMillis {
            withContext(Dispatchers.IO) {
                val m = n.value!!
                val A = genMatrix(m)
                val B = genMatrix(m)
                val C = matMulCPU(A, B, m)
            }
        }
        println(t)
        onCPUStatsChanged(CPUData(t))
    }

    fun cpuBenchmark(){
        viewModelScope.launch(){
            _cpuBenchmark()
        }
    }

}
