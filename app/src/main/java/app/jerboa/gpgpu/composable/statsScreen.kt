package app.jerboa.gpgpu.composable

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import app.jerboa.gpgpu.ViewModel.StatsViewModel
import app.jerboa.gpgpu.data.CPUData
import app.jerboa.gpgpu.data.GPUData

@Composable
fun statsScreen(statsViewModel: StatsViewModel, images: Map<String,Int> = mapOf()){
    val cpuStats: CPUData by statsViewModel.cpuStats.observeAsState(initial = CPUData(0))
    val gpuStats: GPUData by statsViewModel.gpuStats.observeAsState(GPUData(0,0,0,0f))
    val n: Int by statsViewModel.n.observeAsState(2)
    val isWaiting: Boolean by statsViewModel.isWaiting.observeAsState(false)

    stats(
        cpuStats=cpuStats,
        gpuStats=gpuStats,
        n=n,
        isWaiting,
        onCPUStatsChanged = {statsViewModel.onCPUStatsChanged(it)},
        onGPUStatsChanged = {statsViewModel.onGPUStatsChanged(it)},
        onNChanged = {statsViewModel.onNChanged(it)},
        cpuBenchmark = {statsViewModel.cpuBenchmark()},
        gpuBenchmark = {statsViewModel.gpuBenchmark()},
        images = images
    )
}