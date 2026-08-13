import 'package:flutter/material.dart';

class Exp6Theme extends StatelessWidget {
  const Exp6Theme({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Themes and Styles Example',
      theme: ThemeData(
        primaryColor: Colors.blue,
        fontFamily: 'Roboto',
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Custom Theme Example',
        ),
      ),
      body: const Center(
        child: Text(
          'Hello, Flutter!',
          style: TextStyle(
            fontSize: 24,
          ),
        ),
      ),
    );
  }
}