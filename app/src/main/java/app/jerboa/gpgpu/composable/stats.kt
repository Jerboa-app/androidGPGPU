package app.jerboa.gpgpu.composable

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.ExperimentalAnimationApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material.Button
import androidx.compose.material.ButtonDefaults
import androidx.compose.material.Icon
import androidx.compose.material.Text
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.runtime.*
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import app.jerboa.gpgpu.ViewModel.StatsViewModel
import app.jerboa.gpgpu.data.CPUData
import app.jerboa.gpgpu.data.GPUData
import app.jerboa.gpgpu.composable.slider
import app.jerboa.gpgpu.ui.theme.DTriangle


@OptIn(ExperimentalAnimationApi::class)
@Composable
fun stats(
    cpuStats: CPUData,
    gpuStats: GPUData,
    n: Int,
    isWaiting:Boolean,
    onCPUStatsChanged: (CPUData)->Unit,
    onGPUStatsChanged: (GPUData)->Unit,
    onNChanged: (Int)->Unit,
    cpuBenchmark: ()->Unit,
    gpuBenchmark: ()->Unit,
    images: Map<String,Int> = mapOf()
){
    Column(){
        slider("Matrix Size (NxN)",onNChanged)
        Spacer(modifier = Modifier.height(32.dp))
        Row() {
            Text("Runtime: " + cpuStats.time +" ms", Modifier.weight(1f))
            Spacer(modifier = Modifier.height(32.dp))
            Button(onClick = {
                cpuBenchmark()
            }) {
                // Inner content including an icon and a text label
                var theta by remember { mutableStateOf(270f)}
                AnimatedContent(targetState = theta) {
                    Image(
                        painter = painterResource(id = images["logo"]!!),
                        contentDescription = "Image",
                        modifier = Modifier
                            .rotate(270f)
                            .size(40.dp)
                            .alpha(if(isWaiting){0.33f}else{1f})
                            .clip(
                                DTriangle(with(LocalDensity.current) { 32.dp.toPx() })
                            )
                    )
                }
                Spacer(Modifier.size(ButtonDefaults.IconSpacing))
                Text("CPU")

            }
        }
        Spacer(modifier = Modifier.height(64.dp))
        Row() {
            Column(Modifier.weight(1f)) {
                Text("Runtime: " + gpuStats.time + " ms")
                Text("Drawing: " + gpuStats.drawTime+" ns")
                Text("Memory Transfer: " + gpuStats.textureTime+" ms")
                Text("Error: " + gpuStats.error)
            }
            Button(onClick = {
                gpuBenchmark()
            }) {
                // Inner content including an icon and a text label
                var theta by remember { mutableStateOf(270f)}
                AnimatedContent(targetState = theta) {
                    Image(
                        painter = painterResource(id = images["logo"]!!),
                        contentDescription = "Image",
                        modifier = Modifier
                            .rotate(270f)
                            .size(40.dp)
                            .alpha(if(isWaiting){0.33f}else{1f})
                            .clip(
                                DTriangle(with(LocalDensity.current) { 32.dp.toPx() })
                            )
                    )
                }
                Spacer(Modifier.size(ButtonDefaults.IconSpacing))
                Text("GPU")
            }
        }
    }
}