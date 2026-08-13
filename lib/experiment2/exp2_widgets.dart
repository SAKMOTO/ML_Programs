import 'package:flutter/material.dart';

class Exp2Widgets extends StatelessWidget {
  const Exp2Widgets({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
       appBar: AppBar(
        title: const Text('Text Widget Example'),
      ),
      body: const Center(
        child: Text(
          'Hello, Flutter!',
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
    );
  }
}

class Exp2Image extends StatelessWidget {
  const Exp2Image({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Image Widget')),
      body: Image.network('https://picsum.photos/250?image=9'),
    );
  }
}

class Exp2TextAndImage extends StatelessWidget {
  const Exp2TextAndImage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Image Widget')),
      body: Image.asset('assets/images/Harun.jpg'),
    );
  }
}

class Exp2container extends StatelessWidget {
  const Exp2container({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('container Image')),
      body: Center(
        child: Container(
          width: 200,
          height: 200,
          padding: const EdgeInsets.all(16),
          margin: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.blue,
            borderRadius: BorderRadius.circular(8),
            boxShadow: const [
              BoxShadow(
                color: Colors.black26,
                blurRadius: 10,
                offset: Offset(2, 2),
              ),
            ],
          ),
          child: const Center(
            child: Text(
              'Container',
              style: TextStyle(color: Color.fromARGB(255, 0, 0, 0), fontSize: 24),
            ),
          ),
        ),
      ),
    );
  }
}

class Exp2Card extends StatelessWidget {
  const Exp2Card({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('card widget ')),
      body: Center(
        child: Card(
          elevation: 5,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
          child: Container(
            width: 200,
            height: 100,
            padding: EdgeInsets.all(16),
            child: Column(
              children: [
                Text('Card Title'),
                SizedBox(height: 8),
                Text('Card Subtitle'),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
