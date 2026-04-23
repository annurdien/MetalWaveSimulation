#include <metal_stdlib>
using namespace metal;

struct WaveSimulationUniforms {
    uint2 textureSize;
    float propagation;
    float damping;
    float viscosity;
    float dispersion;
    float boundaryReflection;
    float boundaryWidth;
    float deltaTime;
};

struct DisturbanceUniform {
    float2 center;
    float radius;
    float strength;
};

struct RenderUniforms {
    float2 textureSize;
    float time;
    float intensity;
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

kernel void waveStep(
    texture2d<float, access::read> previousTexture [[texture(0)]],
    texture2d<float, access::read> currentTexture [[texture(1)]],
    texture2d<float, access::write> nextTexture [[texture(2)]],
    constant WaveSimulationUniforms& uniforms [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uniforms.textureSize.x || gid.y >= uniforms.textureSize.y) {
        return;
    }

    uint2 left = uint2((gid.x == 0) ? gid.x : gid.x - 1, gid.y);
    uint2 right = uint2(min(gid.x + 1, uniforms.textureSize.x - 1), gid.y);
    uint2 down = uint2(gid.x, (gid.y == 0) ? gid.y : gid.y - 1);
    uint2 up = uint2(gid.x, min(gid.y + 1, uniforms.textureSize.y - 1));

    uint2 upLeft = uint2((gid.x == 0) ? gid.x : gid.x - 1, min(gid.y + 1, uniforms.textureSize.y - 1));
    uint2 upRight = uint2(min(gid.x + 1, uniforms.textureSize.x - 1), min(gid.y + 1, uniforms.textureSize.y - 1));
    uint2 downLeft = uint2((gid.x == 0) ? gid.x : gid.x - 1, (gid.y == 0) ? gid.y : gid.y - 1);
    uint2 downRight = uint2(min(gid.x + 1, uniforms.textureSize.x - 1), (gid.y == 0) ? gid.y : gid.y - 1);

    float current = currentTexture.read(gid).r;
    float previous = previousTexture.read(gid).r;

    float currentL = currentTexture.read(left).r;
    float currentR = currentTexture.read(right).r;
    float currentD = currentTexture.read(down).r;
    float currentU = currentTexture.read(up).r;

    float currentUL = currentTexture.read(upLeft).r;
    float currentUR = currentTexture.read(upRight).r;
    float currentDL = currentTexture.read(downLeft).r;
    float currentDR = currentTexture.read(downRight).r;

    float previousL = previousTexture.read(left).r;
    float previousR = previousTexture.read(right).r;
    float previousD = previousTexture.read(down).r;
    float previousU = previousTexture.read(up).r;

    float laplacian4 = currentL + currentR + currentD + currentU - (4.0 * current);
    float laplacianDiagonal = currentUL + currentUR + currentDL + currentDR - (4.0 * current);
    float laplacian = (4.0 * laplacian4 + laplacianDiagonal) / 6.0;

    float microChop = laplacianDiagonal - laplacian4;

    float velocity = current - previous;
    float velocityNeighbors =
        (currentL - previousL) +
        (currentR - previousR) +
        (currentD - previousD) +
        (currentU - previousU);
    velocityNeighbors *= 0.25;

    float viscosityBlend = clamp(uniforms.viscosity * uniforms.deltaTime * 12.0, 0.0, 0.5);
    velocity = mix(velocity, velocityNeighbors, viscosityBlend);

    float acceleration = uniforms.propagation * (laplacian + uniforms.dispersion * microChop);
    velocity += acceleration;

    float drag = exp(-uniforms.damping * uniforms.deltaTime);
    velocity *= drag;

    float minDimension = float(min(uniforms.textureSize.x, uniforms.textureSize.y));
    float edgeZone = max(1.0, uniforms.boundaryWidth * minDimension);
    float edgeDistance = min(
        min(float(gid.x), float(uniforms.textureSize.x - 1 - gid.x)),
        min(float(gid.y), float(uniforms.textureSize.y - 1 - gid.y))
    );
    float interior = smoothstep(0.0, 1.0, edgeDistance / edgeZone);
    float boundaryAttenuation = mix(uniforms.boundaryReflection, 1.0, interior);
    velocity *= boundaryAttenuation;

    float next = current + velocity;
    next *= boundaryAttenuation;
    nextTexture.write(clamp(next, -2.0, 2.0), gid);
}

kernel void applyDisturbances(
    texture2d<float, access::read_write> currentTexture [[texture(0)]],
    texture2d<float, access::read_write> previousTexture [[texture(1)]],
    constant DisturbanceUniform* disturbances [[buffer(0)]],
    constant uint& disturbanceCount [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = currentTexture.get_width();
    uint height = currentTexture.get_height();

    if (gid.x >= width || gid.y >= height) {
        return;
    }

    float2 uv = (float2(gid) + 0.5) / float2(width, height);
    float impulse = 0.0;

    for (uint i = 0; i < disturbanceCount; ++i) {
        DisturbanceUniform disturbance = disturbances[i];
        float2 delta = uv - disturbance.center;
        float radiusSquared = disturbance.radius * disturbance.radius;
        float distanceSquared = dot(delta, delta);

        if (distanceSquared < radiusSquared) {
            float falloff = exp(-distanceSquared / max(radiusSquared * 0.18, 1e-6));
            impulse += disturbance.strength * falloff;
        }
    }

    if (impulse != 0.0) {
        float current = currentTexture.read(gid).r;
        float previous = previousTexture.read(gid).r;

        float displacement = impulse * 0.18;
        float velocityKick = impulse * 0.82;

        current += displacement + (0.5 * velocityKick);
        previous += displacement - (0.5 * velocityKick);

        currentTexture.write(clamp(current, -2.0, 2.0), gid);
        previousTexture.write(clamp(previous, -2.0, 2.0), gid);
    }
}

vertex VertexOut waveVertex(uint vertexID [[vertex_id]]) {
    constexpr float2 positions[6] = {
        float2(-1.0, -1.0),
        float2(1.0, -1.0),
        float2(-1.0, 1.0),
        float2(1.0, -1.0),
        float2(1.0, 1.0),
        float2(-1.0, 1.0)
    };

    constexpr float2 uvs[6] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 0.0),
        float2(1.0, 1.0),
        float2(0.0, 1.0)
    };

    VertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.uv = uvs[vertexID];
    return out;
}

fragment float4 waveFragment(
    VertexOut in [[stage_in]],
    texture2d<float, access::sample> waveTexture [[texture(0)]],
    constant RenderUniforms& uniforms [[buffer(0)]]
) {
    constexpr sampler pixelSampler(
        coord::normalized,
        address::clamp_to_edge,
        filter::nearest
    );

    float2 texel = 1.0 / uniforms.textureSize;

    float center = waveTexture.sample(pixelSampler, in.uv).r;
    float left = waveTexture.sample(pixelSampler, in.uv - float2(texel.x, 0.0)).r;
    float right = waveTexture.sample(pixelSampler, in.uv + float2(texel.x, 0.0)).r;
    float down = waveTexture.sample(pixelSampler, in.uv - float2(0.0, texel.y)).r;
    float up = waveTexture.sample(pixelSampler, in.uv + float2(0.0, texel.y)).r;

    float slopeX = right - left;
    float slopeY = up - down;
    float slopeMagnitude = length(float2(slopeX, slopeY));
    float curvature = abs(left + right + up + down - (4.0 * center));

    // Compute wave activity to drive alpha — no activity means fully transparent
    float activity = saturate((abs(center) * 6.0) + (slopeMagnitude * 12.0) + (curvature * 8.0));
    if (activity < 1e-4) {
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    float3 normal = normalize(float3(-slopeX * 2.2, -slopeY * 2.2, 1.0));
    float3 lightDirection = normalize(float3(-0.28, 0.62, 0.74));
    float3 viewDirection = normalize(float3(0.0, 0.0, 1.0));
    float3 halfVector = normalize(lightDirection + viewDirection);

    float diffuse = max(dot(normal, lightDirection), 0.0);
    float fresnel = pow(1.0 - saturate(dot(normal, viewDirection)), 4.0);

    float depthMix = saturate(0.5 + center * 0.45);
    float3 deepColor = float3(0.006, 0.055, 0.11);
    float3 shallowColor = float3(0.02, 0.28, 0.4);
    float3 skyColor = float3(0.42, 0.62, 0.79);

    float3 baseWater = mix(deepColor, shallowColor, depthMix);
    float3 reflection = mix(float3(0.09, 0.16, 0.24), skyColor, fresnel);

    float gloss = mix(28.0, 88.0, 1.0 - saturate(curvature * 2.6));
    float specular = pow(max(dot(normal, halfVector), 0.0), gloss) * (0.14 + 0.45 * fresnel);

    float foam = smoothstep(0.22, 0.62, curvature + slopeMagnitude * 0.15);

    float3 color = baseWater * (0.3 + 0.65 * diffuse);
    color += reflection * 0.42;
    color += specular;
    color += foam * float3(0.2, 0.24, 0.25);

    float2 cell = abs(fract(in.uv * uniforms.textureSize) - 0.5);
    float pixelAccent = smoothstep(0.45, 0.5, max(cell.x, cell.y));
    color *= mix(0.98, 1.02, pixelAccent);

    // Smooth alpha from wave activity so ripples fade to transparent
    float alpha = smoothstep(0.0, 0.15, activity);

    return float4(saturate(color) * alpha, alpha);
}
