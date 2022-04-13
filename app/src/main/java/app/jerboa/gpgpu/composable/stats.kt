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

/*
    Composable to actually show the data to the screen.

    Hooks up to the view model for the cpu and gpu benchmark data via cpuStats: CPUData,
    and gpuStats: GPUData (see statsScreen.kt).

    Events are passed up to the ViewModel view onNChanged, gpuBenchmark, cpuBenchmark

    The Optin for experimental animations can be removed from the Button images quite easily
 */
@OptIn(ExperimentalAnimationApi::class)
@Composable
fun stats(
    cpuStats: CPUData,
    gpuStats: GPUData,
    n: Int,
    isWaiting:Pair<Boolean,Boolean>,
    onCPUStatsChanged: (CPUData)->Unit,
    onGPUStatsChanged: (GPUData)->Unit,
    onNChanged: (Int)->Unit,
    cpuBenchmark: ()->Unit,
    gpuBenchmark: ()->Unit,
    images: Map<String,Int> = mapOf()
){
    Column(){
        // pass onNChanged further down the chain to slider
        slider("Matrix Size (NxN)",onNChanged)
        Spacer(modifier = Modifier.height(32.dp))
        Row() {
            // cpu stats, just a runtime
            Text("Runtime: " + cpuStats.time +" ms", Modifier.weight(1f))
            Spacer(modifier = Modifier.height(32.dp))
            Button(onClick = {
                // pass up to view model to request a cpu benchmark
                cpuBenchmark()
            }) {
                // dummy
                var theta by remember { mutableStateOf(270f)}
                // animate a fade to 0.33f alpha to show user benchmark is running (isWaiting)
                AnimatedContent(targetState = theta) {
                    Image(
                        painter = painterResource(id = images["logo"]!!),
                        contentDescription = "Image",
                        modifier = Modifier
                            .rotate(270f)
                            .size(40.dp)
                            .alpha(if(isWaiting.first){0.33f}else{1f})
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
            // gpu stats, also split the time into drawing and data shunting
            Column(Modifier.weight(1f)) {
                Text("Runtime: " + gpuStats.time + " ms")
                Text("Drawing: " + gpuStats.drawTime+" ns")
                Text("Memory Transfer: " + gpuStats.textureTime+" ms")
                Text("Error: " + gpuStats.error)
            }
            Button(onClick = {
                // pass up to view model to request a gpu benchmark
                gpuBenchmark()
            }) {
                // dummy
                var theta by remember { mutableStateOf(270f)}
                // animate a fade to 0.33f alpha to show user benchmark is running (isWaiting)
                AnimatedContent(targetState = theta) {
                    Image(
                        painter = painterResource(id = images["logo"]!!),
                        contentDescription = "Image",
                        modifier = Modifier
                            .rotate(270f)
                            .size(40.dp)
                            .alpha(if(isWaiting.second){0.33f}else{1f})
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