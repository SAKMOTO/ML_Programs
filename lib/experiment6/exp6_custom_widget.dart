import 'package:flutter/material.dart';

class CustomStyledText extends StatelessWidget {
  final String text;
  final Color textColor;
  final double fontSize;

  const CustomStyledText({
    super.key,
    required this.text,
    this.textColor = Colors.black,
    this.fontSize = 16.0,
  });

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: TextStyle(
        color: textColor,
        fontSize: fontSize,
        fontWeight: FontWeight.bold,
      ),
    );
  }
}