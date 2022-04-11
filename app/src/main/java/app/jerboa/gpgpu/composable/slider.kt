package app.jerboa.gpgpu.composable

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material.Slider
import androidx.compose.material.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import app.jerboa.gpgpu.ViewModel.CONSTANTS
import kotlin.math.floor

/*
    Slider composable endowed with a title and status label in a
    column format, part of the StatsViewModel view onNChanged
*/
@Composable
fun slider(title: String, onNChanged: (Int) -> Unit){
    var sliderPosition by remember { mutableStateOf(2f) }
    Column(
        Modifier
            .padding(all = 8.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally) {

        Text(title)
        Text(text = floor(sliderPosition).toInt().toString())
        Slider(
            valueRange = 2f..CONSTANTS.maxN.toFloat(),
            steps = (CONSTANTS.maxN - 2),
            value = sliderPosition,
            onValueChange = {
                sliderPosition = it
                onNChanged(floor(it).toInt())
            }
        )
    }
}