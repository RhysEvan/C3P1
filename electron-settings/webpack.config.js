const path = require("path");

module.exports = {
    mode: "development",
    entry: './src/react/index.js',
    module: {
        rules: [
            {
                test: /\.jsx?$/,
                exclude: /node_modules/,
                use: {
                    loader: 'swc-loader',
                    options: {
                        jsc: {
                            parser: {
                                syntax: 'ecmascript',
                                jsx: true,
                            },
                            target: 'es2015',
                            loose: true,
                        },
                    },
                },
            },
            {
                test: /\.css$/,
                use: ['style-loader', 'css-loader', 'postcss-loader']
            }
        ],
    },
    resolve: {
        extensions: ['.js', '.jsx'],
        alias: {
            "@": path.resolve(__dirname, "src"),
        },
    },
    output: {
        filename: 'bundle.js',
        path: path.resolve(__dirname, 'dist'),
    },
};