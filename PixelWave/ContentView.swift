//
//  ContentView.swift
//  PixelWave
//
//  Created by Annurdien Rasyid on 23/04/26.
//

import SwiftUI
import Metal

struct ContentView: View {
    @State private var parameters = WaveParameters()
    @State private var resetCount = 0
    @State private var showControl = true

    private let isMetalAvailable = MTLCreateSystemDefaultDevice() != nil

    var body: some View {
        Group {
            if isMetalAvailable {
                simulationLayout
            } else {
                unsupportedView
            }
        }
    }

    private var simulationLayout: some View {
        ZStack(alignment: .topLeading) {
            WaveSimulationView(parameters: $parameters, resetCount: resetCount)
                .background(.white)
                .ignoresSafeArea()

            controlPanel
        }
        .background(Color.black)
    }

    private var unsupportedView: some View {
        VStack(spacing: 12) {
            Image(systemName: "xmark.circle")
                .font(.system(size: 38, weight: .semibold))
            Text("Metal is unavailable on this device")
                .font(.headline)
            Text("Pixel Wave requires Metal support to run the simulation.")
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
        }
        .padding(24)
    }

    @ViewBuilder
    private var controlPanel: some View {
        if !showControl {
            HStack {
                Button("Control") {
                    withAnimation {
                        showControl = true
                    }
                }
                .buttonStyle(.glassProminent)
               
                
                Spacer()
                
                Button("Reset Surface") {
                    resetCount += 1
                }
                .buttonStyle(.glassProminent)
            }
            .padding()
            
        } else {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Pixel Wave")
                        .font(.system(.title2, design: .rounded).weight(.bold))
                    Spacer()
                    Button("X") {
                        withAnimation {
                            showControl = false
                        }
                    }
                    .buttonStyle(.glassProminent)
                    
                }

                Text("Tap or drag to inject ripples.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                Divider()

                ParameterSlider(
                    title: "Wave Speed",
                    value: $parameters.waveSpeed,
                    range: 6.0...32.0,
                    decimals: 1
                )

                ParameterSlider(
                    title: "Damping",
                    value: $parameters.damping,
                    range: 0.05...2.2,
                    decimals: 2
                )

                ParameterSlider(
                    title: "Viscosity",
                    value: $parameters.viscosity,
                    range: 0.0...1.2,
                    decimals: 2
                )

                ParameterSlider(
                    title: "Surface Tension",
                    value: $parameters.dispersion,
                    range: 0.0...1.0,
                    decimals: 2
                )

                ParameterSlider(
                    title: "Edge Reflection",
                    value: $parameters.edgeReflection,
                    range: 0.1...1.0,
                    decimals: 2
                )

                ParameterSlider(
                    title: "Shore Width",
                    value: $parameters.edgeWidth,
                    range: 0.01...0.2,
                    decimals: 3
                )

                ParameterSlider(
                    title: "Brush Radius",
                    value: $parameters.brushRadius,
                    range: 0.01...0.08,
                    decimals: 3
                )

                ParameterSlider(
                    title: "Impulse",
                    value: $parameters.impulse,
                    range: 0.1...1.2,
                    decimals: 2
                )
                Divider()

                Text("Wave Colors")
                    .font(.caption.weight(.semibold))

                HStack(spacing: 12) {
                    ColorPicker("Deep", selection: colorBinding(for: \.deepColor), supportsOpacity: false)
                        .font(.caption)
                    ColorPicker("Mid", selection: colorBinding(for: \.shallowColor), supportsOpacity: false)
                        .font(.caption)
                    ColorPicker("Sky", selection: colorBinding(for: \.skyColor), supportsOpacity: false)
                        .font(.caption)
                }

                HStack {
                    Button("Reset Surface") {
                        resetCount += 1
                    }
                    .buttonStyle(.borderedProminent)

                    Button("Defaults") {
                        withAnimation {
                            parameters = WaveParameters()
                        }
                        resetCount += 1
                    }
                    .buttonStyle(.bordered)

                    Spacer()

                    Text("120 Hz")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(16)
            .frame(maxWidth: 340)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .padding(.horizontal, 16)
            .padding(.top, 16)
        }
    }

    private func colorBinding(for keyPath: WritableKeyPath<WaveParameters, SIMD3<Float>>) -> Binding<Color> {
        Binding(
            get: {
                let v = parameters[keyPath: keyPath]
                return Color(red: Double(v.x), green: Double(v.y), blue: Double(v.z))
            },
            set: { newColor in
                let uiColor = UIColor(newColor)
                var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
                uiColor.getRed(&r, green: &g, blue: &b, alpha: &a)
                parameters[keyPath: keyPath] = SIMD3<Float>(Float(r), Float(g), Float(b))
            }
        )
    }
}

private struct ParameterSlider: View {
    let title: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    let decimals: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title)
                    .font(.caption.weight(.semibold))
                Spacer()
                Text(value, format: .number.precision(.fractionLength(decimals)))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            Slider(
                value: Binding(
                    get: { Double(value) },
                    set: { value = Float($0) }
                ),
                in: Double(range.lowerBound)...Double(range.upperBound)
            )
        }
    }
}

#Preview {
    ContentView()
}
