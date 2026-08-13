import 'package:flutter/material.dart';

class Exp7Validation extends StatefulWidget {
  const Exp7Validation({super.key});

  @override
  State<Exp7Validation> createState() =>
      _Exp7ValidationState();
}

class _Exp7ValidationState
    extends State<Exp7Validation> {
  final _formKey = GlobalKey<FormState>();

  String _name = '';
  String _email = '';
  String _password = '';

  void _submitForm() {
    if (_formKey.currentState!.validate()) {
      _formKey.currentState!.save();

      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text("Success"),
          content: Text(
            "Form Submitted!\n\n"
            "Name: $_name\n"
            "Email: $_email",
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(context);
              },
              child: const Text("OK"),
            ),
          ],
        ),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            "Please fix the errors in red",
          ),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Form Validation',
        ),
      ),

      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,

          child: Column(
            crossAxisAlignment:
                CrossAxisAlignment.start,

            children: [

              // Name
              TextFormField(
                decoration: const InputDecoration(
                  labelText: 'Name',
                  border: OutlineInputBorder(),
                ),

                validator: (value) {
                  if (value == null ||
                      value.trim().isEmpty) {
                    return 'Name is required';
                  }

                  return null;
                },

                onSaved: (value) {
                  _name = value!;
                },
              ),

              const SizedBox(height: 16),

              // Email
              TextFormField(
                decoration: const InputDecoration(
                  labelText: 'Email',
                  border: OutlineInputBorder(),
                ),

                keyboardType:
                    TextInputType.emailAddress,

                validator: (value) {
                  if (value == null ||
                      value.trim().isEmpty) {
                    return 'Email is required';
                  }

                  if (!RegExp(
                    r'^[^@]+@[^@]+\.[^@]+',
                  ).hasMatch(value)) {
                    return 'Enter a valid email';
                  }

                  return null;
                },

                onSaved: (value) {
                  _email = value!;
                },
              ),

              const SizedBox(height: 16),

              // Password
              TextFormField(
                decoration: const InputDecoration(
                  labelText: 'Password',
                  border: OutlineInputBorder(),
                ),

                obscureText: true,

                validator: (value) {
                  if (value == null ||
                      value.length < 6) {
                    return 'Password must be at least 6 characters';
                  }

                  return null;
                },

                onSaved: (value) {
                  _password = value!;
                },
              ),

              const SizedBox(height: 20),

              // Submit
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _submitForm,
                  child: const Text('Submit'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}