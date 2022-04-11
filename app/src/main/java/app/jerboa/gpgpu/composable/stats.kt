package app.jerboa.gpgpu.composable

import androidx.compose.foundation.layout.*
import androidx.compose.material.Button
import androidx.compose.material.ButtonDefaults
import androidx.compose.material.Icon
import androidx.compose.material.Text
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import app.jerboa.gpgpu.ViewModel.StatsViewModel
import app.jerboa.gpgpu.data.CPUData
import app.jerboa.gpgpu.data.GPUData
import app.jerboa.gpgpu.composable.slider

@Composable
fun stats(
    cpuStats: CPUData,
    gpuStats: GPUData,
    n: Int,
    onCPUStatsChanged: (CPUData)->Unit,
    onGPUStatsChanged: (GPUData)->Unit,
    onNChanged: (Int)->Unit,
    cpuBenchmark: ()->Unit
){
    Column(){
        slider("Matrix Size (NxN)",onNChanged)
        Spacer(modifier = Modifier.height(32.dp))
        Row() {
            Text("Runtime: " + cpuStats.time, Modifier.weight(1f))
            Spacer(modifier = Modifier.height(32.dp))
            Button(onClick = {
                cpuBenchmark()
            }) {
                // Inner content including an icon and a text label
                Icon(
                    Icons.Filled.Favorite,
                    contentDescription = "Favorite",
                    modifier = Modifier.size(ButtonDefaults.IconSize)
                )
                Spacer(Modifier.size(ButtonDefaults.IconSpacing))
                Text("Like")

            }
        }
        Column(Modifier.weight(1f)) {
            Text("Runtime: " + gpuStats.time)
            Text("Drawing: " + gpuStats.drawTime)
            Text("Memory Transfer: " + gpuStats.textureTime)
            Text("Error: " + gpuStats.error)
        }
    }
}