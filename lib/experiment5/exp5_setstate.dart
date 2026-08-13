import 'package:flutter/material.dart';

class Exp5SetState extends StatefulWidget {
  const Exp5SetState({super.key});

  @override
  State<Exp5SetState> createState() =>
      _Exp5SetStateState();
}

class _Exp5SetStateState
    extends State<Exp5SetState> {
  int count = 0;

  void increment() {
    setState(() {
      count++;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('setState Example'),
      ),
      body: Center(
        child: Text(
          'Count: $count',
          style: const TextStyle(
            fontSize: 24,
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: increment,
        child: const Icon(Icons.add),
      ),
    );
  }
}