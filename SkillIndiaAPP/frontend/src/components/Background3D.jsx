import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Stars, Sparkles } from '@react-three/drei';

function RotatingStars() {
    const ref = useRef();
    useFrame((state, delta) => {
        ref.current.rotation.x -= delta / 15;
        ref.current.rotation.y -= delta / 20;
    });
    return (
        <group ref={ref}>
            <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
        </group>
    );
}

export default function Background3D() {
    return (
        <div className="fixed inset-0 z-0 pointer-events-none bg-black">
            <Canvas camera={{ position: [0, 0, 1] }}>
                <RotatingStars />
                {/* Vampire Red Sparkles */}
                <Sparkles
                    count={150}
                    scale={12}
                    size={4}
                    speed={0.4}
                    opacity={0.6}
                    color="#D32F2F"
                />
                {/* Blue Sparkles */}
                <Sparkles
                    count={150}
                    scale={12}
                    size={4}
                    speed={0.4}
                    opacity={0.6}
                    color="#3B82F6"
                />
            </Canvas>
        </div>
    );
}
