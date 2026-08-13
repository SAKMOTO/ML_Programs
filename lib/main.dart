import 'package:flutter/material.dart';
import 'experiment2/exp2_widgets.dart';
import 'experiment2/exp2_layouts.dart';
import 'experiment2/exp2_counter.dart';
import 'experiment3/exp3_responsive.dart';
import 'experiment3/exp3_breakpoints.dart';
import 'experiment4/exp4_navigation.dart';
import 'experiment4/exp4_named_routes.dart';
import 'experiment5/exp5_stateful_stateless.dart';
import 'experiment5/exp5_setstate.dart';
import 'experiment6/exp6_custom_widget.dart';
import 'experiment6/exp6_theme.dart';
import 'experiment7/exp7_form.dart';
import 'experiment7/exp7_validation.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Exp7Validation(), 
      );
  }
}



