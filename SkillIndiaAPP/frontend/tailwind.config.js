/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                vampire: {
                    black: '#050505', // Deeper black
                    dark: '#121212',
                    red: '#D32F2F', // More vibrant, Apple-like red
                    blood: '#FF5252',
                    gray: '#E0E0E0',
                    glass: 'rgba(255, 255, 255, 0.05)', // Glass effect base
                }
            },
            fontFamily: {
                sans: ['"Inter"', 'sans-serif'], // Clean, modern font
            },
            backdropBlur: {
                xs: '2px',
            }
        },
    },
    plugins: [],
}
