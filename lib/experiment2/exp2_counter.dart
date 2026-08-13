import 'package:flutter/material.dart';

class Exp2Counter extends StatefulWidget {
  const Exp2Counter({super.key});

  @override
  State<Exp2Counter> createState() => _Exp2CounterState();
}

class _Exp2CounterState extends State<Exp2Counter> {
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