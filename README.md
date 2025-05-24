# SafeStreet Application

This is a full-stack application that uses React for the frontend and Express/Node.js with MongoDB for the backend.

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- MongoDB (local installation or MongoDB Atlas account)

### Installation

1. Clone the repository
2. Install dependencies:

```bash
cd firstapp
npm install
```

### Setting Up MongoDB

By default, the application will attempt to connect to a MongoDB instance at `mongodb://localhost:27017/safestreet`. You can change this by creating a `.env` file in the `Backend` directory:

```
MONGODB_URI=mongodb://localhost:27017/safestreet
```

If you're using MongoDB Atlas or a different MongoDB connection string, update this value accordingly.

### Creating Test Users

Before logging in, you need to create test users in the MongoDB database:

```bash
npm run create-user
```

This will create:
- A regular user with email: `user@example.com` and password: `password123`
- An admin user with email: `admin@safestreet.com` and password: `admin123`

### Running the Application

1. Start the backend API server:

```bash
npm run api
```

2. In a separate terminal, start the frontend:

```bash
npm run dev
```

3. Open your browser and navigate to http://localhost:3000

## Fixing Authentication Issues

If you experience authentication issues:

1. Make sure MongoDB is running and accessible
2. Check that the MongoDB connection string in `.env` is correct
3. Make sure test users are created with `npm run create-user`
4. Verify that the authorization header is being sent correctly

### Basic Authentication Flow

The application uses Basic authentication directly with MongoDB:

1. User enters email/password on the login page
2. Credentials are stored in localStorage and sent with each API request
3. Backend validates the credentials against MongoDB
4. If valid, the user details are returned and stored

### Troubleshooting

- **500 Server Error**: Check if MongoDB is running and properly connected
- **401 Unauthorized**: Credentials are incorrect or not being sent properly
- **Failed to load images**: User authentication may have failed

### Default Users

- Regular User:
  - Email: user@example.com
  - Password: password123

- Admin User:
  - Email: admin@safestreet.com
  - Password: admin123

## Features

- User authentication with MongoDB credentials
- Image upload and classification
- Admin dashboard for image management

# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
