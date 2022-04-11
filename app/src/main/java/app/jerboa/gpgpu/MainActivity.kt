package app.jerboa.gpgpu

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.*
import androidx.compose.material.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

import app.jerboa.gpgpu.ViewModel.StatsViewModel
import app.jerboa.gpgpu.composable.statsScreen
import app.jerboa.gpgpu.ui.theme.GPGPUTheme

class MainActivity : ComponentActivity() {

    // viewmodels
    private val statsViewModel by viewModels<StatsViewModel>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            GPGPUTheme() {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colors.background
                ) {
                    // main column
                    Column(Modifier.padding(all=8.dp)) {
                        // the screen!
                        statsScreen(statsViewModel)
                    }
                }
            }
        }
    }
}