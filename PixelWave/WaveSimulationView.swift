import SwiftUI
import MetalKit

struct WaveParameters: Equatable {
    var waveSpeed: Float = 18.0
    var damping: Float = 0.75
    var viscosity: Float = 0.35
    var dispersion: Float = 0.24
    var edgeReflection: Float = 0.5
    var edgeWidth: Float = 0.08
    var brushRadius: Float = 0.025
    var impulse: Float = 0.62
    var pixelSize: Float = 3.0
}

struct WaveSimulationView: UIViewRepresentable {
    @Binding var parameters: WaveParameters
    var resetCount: Int

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> InteractiveMTKView {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return InteractiveMTKView(frame: .zero, device: nil)
        }

        let view = InteractiveMTKView(frame: .zero, device: device)
        view.translatesAutoresizingMaskIntoConstraints = false
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        view.preferredFramesPerSecond = 120
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = false
        view.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)
        view.isOpaque = false
        view.backgroundColor = .clear
        view.layer.isOpaque = false

        if let renderer = WaveRenderer(device: device, view: view, parameters: parameters) {
            context.coordinator.renderer = renderer
            view.delegate = renderer
            view.onInteraction = { normalizedPoint, pressure in
                renderer.enqueueDisturbance(
                    center: normalizedPoint,
                    radius: parameters.brushRadius,
                    strength: parameters.impulse * pressure
                )
            }
        }

        return view
    }

    func updateUIView(_ uiView: InteractiveMTKView, context: Context) {
        guard let renderer = context.coordinator.renderer else {
            return
        }

        renderer.update(parameters: parameters)
        uiView.onInteraction = { normalizedPoint, pressure in
            renderer.enqueueDisturbance(
                center: normalizedPoint,
                radius: parameters.brushRadius,
                strength: parameters.impulse * pressure
            )
        }

        if context.coordinator.lastResetCount != resetCount {
            renderer.resetSurface()
            context.coordinator.lastResetCount = resetCount
        }
    }

    final class Coordinator {
        var renderer: WaveRenderer?
        var lastResetCount = 0
    }
}

final class InteractiveMTKView: MTKView {
    var onInteraction: ((SIMD2<Float>, Float) -> Void)?

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesBegan(touches, with: event)
        handleTouches(touches)
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesMoved(touches, with: event)
        handleTouches(touches)
    }

    private func handleTouches(_ touches: Set<UITouch>) {
        guard bounds.width > 0, bounds.height > 0 else {
            return
        }

        for touch in touches {
            let location = touch.location(in: self)
            let normalizedX = clamp01(Float(location.x / bounds.width))
            let normalizedY = clamp01(Float(1.0 - (location.y / bounds.height)))

            let intensity: Float
            if touch.maximumPossibleForce > 0 {
                intensity = max(0.2, Float(touch.force / touch.maximumPossibleForce))
            } else {
                intensity = 1.0
            }

            onInteraction?(SIMD2<Float>(normalizedX, normalizedY), intensity)
        }
    }

    private func clamp01(_ value: Float) -> Float {
        min(max(value, 0.0), 1.0)
    }
}
