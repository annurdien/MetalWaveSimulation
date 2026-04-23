import Foundation
import MetalKit
import QuartzCore

private struct WaveSimulationUniforms {
    var textureSize: SIMD2<UInt32>
    var propagation: Float
    var damping: Float
    var viscosity: Float
    var dispersion: Float
    var boundaryReflection: Float
    var boundaryWidth: Float
    var deltaTime: Float
}

private struct DisturbanceUniform {
    var center: SIMD2<Float>
    var radius: Float
    var strength: Float
}

private struct RenderUniforms {
    var textureSize: SIMD2<Float>
    var time: Float
    var intensity: Float
    var deepColor: SIMD3<Float>
    var shallowColor: SIMD3<Float>
    var skyColor: SIMD3<Float>
}

final class WaveRenderer: NSObject, MTKViewDelegate {
    private enum DisturbanceStrategy {
        case readWrite
        case splitPass
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let wavePipeline: MTLComputePipelineState
    private let disturbancePipeline: MTLComputePipelineState
    private let disturbanceStrategy: DisturbanceStrategy
    private let renderPipeline: MTLRenderPipelineState

    private var parameters: WaveParameters
    private var previousTexture: MTLTexture?
    private var currentTexture: MTLTexture?
    private var nextTexture: MTLTexture?

    private var pendingDisturbances: [DisturbanceUniform] = []
    private let disturbanceLock = NSLock()

    private let fixedTimeStep: Float = 1.0 / 120.0
    private let maxSimulationStepsPerFrame = 10

    private var accumulator: Float = 0.0
    private var simulationTime: Float = 0.0
    private var lastFrameTimestamp: CFTimeInterval?

    init?(device: MTLDevice, view: MTKView, parameters: WaveParameters) {
        let disturbanceStrategy: DisturbanceStrategy
        let disturbanceFunctionName: String
        #if targetEnvironment(simulator)
        disturbanceStrategy = .splitPass
        disturbanceFunctionName = "applyDisturbancesSingle"
        #else
        if device.readWriteTextureSupport == .tier2 {
            disturbanceStrategy = .readWrite
            disturbanceFunctionName = "applyDisturbancesReadWrite"
        } else {
            disturbanceStrategy = .splitPass
            disturbanceFunctionName = "applyDisturbancesSingle"
        }
        #endif

        guard
            let commandQueue = device.makeCommandQueue(),
            let library = device.makeDefaultLibrary(),
            let waveFunction = library.makeFunction(name: "waveStep"),
            let disturbanceFunction = library.makeFunction(name: disturbanceFunctionName),
            let vertexFunction = library.makeFunction(name: "waveVertex"),
            let fragmentFunction = library.makeFunction(name: "waveFragment")
        else {
            return nil
        }

        self.device = device
        self.commandQueue = commandQueue
        self.disturbanceStrategy = disturbanceStrategy
        self.parameters = parameters

        do {
            self.wavePipeline = try device.makeComputePipelineState(function: waveFunction)
            self.disturbancePipeline = try device.makeComputePipelineState(function: disturbanceFunction)

            let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
            renderPipelineDescriptor.vertexFunction = vertexFunction
            renderPipelineDescriptor.fragmentFunction = fragmentFunction
            renderPipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
            renderPipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            renderPipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
            renderPipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            renderPipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
            renderPipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            self.renderPipeline = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        } catch {
            return nil
        }

        super.init()

        resizeTexturesIfNeeded(to: view.drawableSize)
    }

    func update(parameters: WaveParameters) {
        self.parameters = parameters
    }

    func enqueueDisturbance(center: SIMD2<Float>, radius: Float, strength: Float) {
        let clampedCenter = SIMD2<Float>(
            min(max(center.x, 0.0), 1.0),
            min(max(center.y, 0.0), 1.0)
        )

        let disturbance = DisturbanceUniform(
            center: clampedCenter,
            radius: max(radius, 0.001),
            strength: strength
        )

        disturbanceLock.lock()
        if pendingDisturbances.count > 64 {
            pendingDisturbances.removeFirst(pendingDisturbances.count - 64)
        }
        pendingDisturbances.append(disturbance)
        disturbanceLock.unlock()
    }

    func resetSurface() {
        guard let previousTexture, let currentTexture, let nextTexture else {
            return
        }

        clear(texture: previousTexture)
        clear(texture: currentTexture)
        clear(texture: nextTexture)

        accumulator = 0.0
        simulationTime = 0.0
        lastFrameTimestamp = nil
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        resizeTexturesIfNeeded(to: size)
    }

    func draw(in view: MTKView) {
        resizeTexturesIfNeeded(to: view.drawableSize)

        guard
            let previousTexture,
            let simulationTexture = currentTexture,
            nextTexture != nil,
            let commandBuffer = commandQueue.makeCommandBuffer(),
            let renderPassDescriptor = view.currentRenderPassDescriptor,
            let drawable = view.currentDrawable
        else {
            return
        }

        let now = CACurrentMediaTime()
        if let lastFrameTimestamp {
            let frameDelta = min(Float(now - lastFrameTimestamp), 1.0 / 20.0)
            accumulator += frameDelta
        }
        lastFrameTimestamp = now

        let disturbances = consumePendingDisturbances(maxCount: 32)
        if !disturbances.isEmpty {
            switch disturbanceStrategy {
            case .readWrite:
                encodeDisturbanceReadWritePass(
                    commandBuffer: commandBuffer,
                    targetTexture: simulationTexture,
                    previousTexture: previousTexture,
                    disturbances: disturbances
                )
            case .splitPass:
                guard let scratchTexture = nextTexture else {
                    return
                }
                encodeDisturbanceSplitPass(
                    commandBuffer: commandBuffer,
                    targetTexture: simulationTexture,
                    previousTexture: previousTexture,
                    scratchTexture: scratchTexture,
                    disturbances: disturbances
                )
            }
        }

        var simulatedSteps = 0
        while accumulator >= fixedTimeStep && simulatedSteps < maxSimulationStepsPerFrame {
            encodeWaveStep(commandBuffer: commandBuffer)
            rotateTextures()
            accumulator -= fixedTimeStep
            simulationTime += fixedTimeStep
            simulatedSteps += 1
        }

        if simulatedSteps == maxSimulationStepsPerFrame {
            accumulator = 0.0
        }

        guard let displayTexture = self.currentTexture else {
            return
        }

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }

        renderEncoder.setRenderPipelineState(renderPipeline)
        renderEncoder.setFragmentTexture(displayTexture, index: 0)

        var renderUniforms = RenderUniforms(
            textureSize: SIMD2<Float>(Float(displayTexture.width), Float(displayTexture.height)),
            time: simulationTime,
            intensity: 1.0,
            deepColor: parameters.deepColor,
            shallowColor: parameters.shallowColor,
            skyColor: parameters.skyColor
        )
        renderEncoder.setFragmentBytes(&renderUniforms, length: MemoryLayout<RenderUniforms>.stride, index: 0)

        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        renderEncoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    private func encodeWaveStep(commandBuffer: MTLCommandBuffer) {
        guard
            let previousTexture,
            let currentTexture,
            let nextTexture,
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            return
        }

        let cfl = parameters.waveSpeed * fixedTimeStep
        let propagation = min(cfl * cfl, 0.48)

        var simulationUniforms = WaveSimulationUniforms(
            textureSize: SIMD2<UInt32>(UInt32(currentTexture.width), UInt32(currentTexture.height)),
            propagation: propagation,
            damping: parameters.damping,
            viscosity: parameters.viscosity,
            dispersion: parameters.dispersion,
            boundaryReflection: parameters.edgeReflection,
            boundaryWidth: parameters.edgeWidth,
            deltaTime: fixedTimeStep
        )

        encoder.setComputePipelineState(wavePipeline)
        encoder.setTexture(previousTexture, index: 0)
        encoder.setTexture(currentTexture, index: 1)
        encoder.setTexture(nextTexture, index: 2)
        encoder.setBytes(
            &simulationUniforms,
            length: MemoryLayout<WaveSimulationUniforms>.stride,
            index: 0
        )

        dispatch(
            encoder: encoder,
            pipeline: wavePipeline,
            width: currentTexture.width,
            height: currentTexture.height
        )

        encoder.endEncoding()
    }

    private func encodeDisturbanceReadWritePass(
        commandBuffer: MTLCommandBuffer,
        targetTexture: MTLTexture,
        previousTexture: MTLTexture,
        disturbances: [DisturbanceUniform]
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        let mutableDisturbances = disturbances
        var disturbanceCount = UInt32(disturbances.count)

        encoder.setComputePipelineState(disturbancePipeline)
        encoder.setTexture(targetTexture, index: 0)
        encoder.setTexture(previousTexture, index: 1)
        mutableDisturbances.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else {
                return
            }
            encoder.setBytes(baseAddress, length: rawBuffer.count, index: 0)
        }
        encoder.setBytes(
            &disturbanceCount,
            length: MemoryLayout<UInt32>.stride,
            index: 1
        )

        dispatch(
            encoder: encoder,
            pipeline: disturbancePipeline,
            width: targetTexture.width,
            height: targetTexture.height
        )

        encoder.endEncoding()
    }

    private func encodeDisturbanceSplitPass(
        commandBuffer: MTLCommandBuffer,
        targetTexture: MTLTexture,
        previousTexture: MTLTexture,
        scratchTexture: MTLTexture,
        disturbances: [DisturbanceUniform]
    ) {
        guard let currentEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }

        let mutableDisturbances = disturbances
        var disturbanceCount = UInt32(disturbances.count)
        var currentScale: Float = 0.59

        currentEncoder.setComputePipelineState(disturbancePipeline)
        currentEncoder.setTexture(targetTexture, index: 0)
        currentEncoder.setTexture(scratchTexture, index: 1)
        mutableDisturbances.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else {
                return
            }
            currentEncoder.setBytes(baseAddress, length: rawBuffer.count, index: 0)
        }
        currentEncoder.setBytes(
            &disturbanceCount,
            length: MemoryLayout<UInt32>.stride,
            index: 1
        )
        currentEncoder.setBytes(
            &currentScale,
            length: MemoryLayout<Float>.stride,
            index: 2
        )

        dispatch(
            encoder: currentEncoder,
            pipeline: disturbancePipeline,
            width: targetTexture.width,
            height: targetTexture.height
        )
        currentEncoder.endEncoding()

        guard let blitCurrent = commandBuffer.makeBlitCommandEncoder() else {
            return
        }
        blitCurrent.copy(
            from: scratchTexture,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: targetTexture.width, height: targetTexture.height, depth: 1),
            to: targetTexture,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blitCurrent.endEncoding()

        guard let previousEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        var previousScale: Float = -0.23
        previousEncoder.setComputePipelineState(disturbancePipeline)
        previousEncoder.setTexture(previousTexture, index: 0)
        previousEncoder.setTexture(scratchTexture, index: 1)
        mutableDisturbances.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else {
                return
            }
            previousEncoder.setBytes(baseAddress, length: rawBuffer.count, index: 0)
        }
        previousEncoder.setBytes(
            &disturbanceCount,
            length: MemoryLayout<UInt32>.stride,
            index: 1
        )
        previousEncoder.setBytes(
            &previousScale,
            length: MemoryLayout<Float>.stride,
            index: 2
        )

        dispatch(
            encoder: previousEncoder,
            pipeline: disturbancePipeline,
            width: previousTexture.width,
            height: previousTexture.height
        )
        previousEncoder.endEncoding()

        guard let blitPrevious = commandBuffer.makeBlitCommandEncoder() else {
            return
        }
        blitPrevious.copy(
            from: scratchTexture,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: previousTexture.width, height: previousTexture.height, depth: 1),
            to: previousTexture,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blitPrevious.endEncoding()
    }

    private func dispatch(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        width: Int,
        height: Int
    ) {
        guard width > 0, height > 0 else {
            return
        }

        let executionWidth = max(1, pipeline.threadExecutionWidth)
        let maxThreads = max(1, pipeline.maxTotalThreadsPerThreadgroup)
        let deviceLimit = device.maxThreadsPerThreadgroup

        // Use a conservative 1D threadgroup for simulator/device compatibility.
        let maxThreadgroupWidth = max(1, min(maxThreads, max(1, deviceLimit.width)))
        let threadgroupWidth = min(executionWidth, maxThreadgroupWidth)

        let threadgroupHeight = 1

        let threadsPerThreadgroup = MTLSize(
            width: threadgroupWidth,
            height: threadgroupHeight,
            depth: 1
        )
        let threadgroupsPerGrid = MTLSize(
            width: (width + threadgroupWidth - 1) / threadgroupWidth,
            height: height,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    private func rotateTextures() {
        let oldPrevious = previousTexture
        previousTexture = currentTexture
        currentTexture = nextTexture
        nextTexture = oldPrevious
    }

    private func consumePendingDisturbances(maxCount: Int) -> [DisturbanceUniform] {
        disturbanceLock.lock()
        defer { disturbanceLock.unlock() }

        guard !pendingDisturbances.isEmpty else {
            return []
        }

        let output: [DisturbanceUniform]
        if pendingDisturbances.count > maxCount {
            output = Array(pendingDisturbances.suffix(maxCount))
        } else {
            output = pendingDisturbances
        }

        pendingDisturbances.removeAll(keepingCapacity: true)
        return output
    }

    private func resizeTexturesIfNeeded(to drawableSize: CGSize) {
        guard drawableSize.width > 0.0, drawableSize.height > 0.0 else {
            return
        }

        let gridWidth = max(64, Int(drawableSize.width / CGFloat(parameters.pixelSize)))
        let gridHeight = max(64, Int(drawableSize.height / CGFloat(parameters.pixelSize)))

        if currentTexture?.width == gridWidth && currentTexture?.height == gridHeight {
            return
        }

        previousTexture = makeWaveTexture(width: gridWidth, height: gridHeight)
        currentTexture = makeWaveTexture(width: gridWidth, height: gridHeight)
        nextTexture = makeWaveTexture(width: gridWidth, height: gridHeight)

        if let previousTexture {
            clear(texture: previousTexture)
        }
        if let currentTexture {
            clear(texture: currentTexture)
        }
        if let nextTexture {
            clear(texture: nextTexture)
        }

        accumulator = 0.0
        simulationTime = 0.0
    }

    private func makeWaveTexture(width: Int, height: Int) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared

        return device.makeTexture(descriptor: descriptor)
    }

    private func clear(texture: MTLTexture) {
        let zeroes = [Float](repeating: 0.0, count: texture.width * texture.height)
        zeroes.withUnsafeBytes { rawBuffer in
            guard let bytes = rawBuffer.baseAddress else {
                return
            }
            texture.replace(
                region: MTLRegionMake2D(0, 0, texture.width, texture.height),
                mipmapLevel: 0,
                withBytes: bytes,
                bytesPerRow: texture.width * MemoryLayout<Float>.stride
            )
        }
    }
}
