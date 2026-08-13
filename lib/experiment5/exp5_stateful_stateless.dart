import 'package:flutter/material.dart';

class Exp5StatefulStateless extends StatelessWidget {
  const Exp5StatefulStateless({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cards Example'),
      ),
      body: const CardList(),
    );
  }
}

class CardList extends StatelessWidget {
  const CardList({super.key});

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      itemCount: 10,
      itemBuilder: (context, index) {
        return CardItem(
          title: 'Card $index',
          subtitle: 'Subtitle $index',
        );
      },
    );
  }
}

class CardItem extends StatelessWidget {
  final String title;
  final String subtitle;

  const CardItem({
    super.key,
    required this.title,
    required this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(
        horizontal: 16,
        vertical: 8,
      ),
      child: ListTile(
        title: Text(title),
        subtitle: Text(subtitle),
        leading: CircleAvatar(
          child: Text(
            title.substring(0, 1),
          ),
        ),
        onTap: () {
          // Handle card tap
        },
      ),
    );
  }
}